"""
    SPG using a single nn-module and a MLP for the distribution params
    
    @Author Leroy Bird
    
    You might need to set, as TF and JAX will both try grab the full gpu memory.
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

import sys
import yaml 
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('AGG')

from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import optax
from jax import random
import numpy as np
from ml_collections import ConfigDict
from jax.experimental.host_callback import id_print
import wandb

from spg import data_loader, data_utils,  spg_dist
from bslibs.plot.qqplot import qqplot



# Global flag to set a specific platform, must be used at startup.
#jax.config.update('jax_platform_name', 'cpu')

def make_qq(preds: list, targets: list, epoch: int, log, averages=[1, 8, 24], output_folder='./results/plots'):
    
    _, axes = plt.subplots(1, len(averages), figsize=(len(averages)*8, 8))
    
    for ax, av in zip(axes, averages):
        # calculate a n-day average.
        def rolling(arr):
            filt = np.ones(av, dtype=np.float32)
            return np.convolve(arr, filt, mode='valid')
        ax.set_title(f'{av} average')
        qqplot(rolling(targets), rolling(preds), ax=ax, num_highlighted_quantiles=10, linewidth=0)
    
    output_path = Path(output_folder) / f'qq_{str(epoch).zfill(3)}.png'
    plt.savefig(output_path, transparent=False, dpi=150)
    plt.close()

    log({'qq_plot' : wandb.Image(str(output_path))})

def save_params(params, epoch : int, output_folder='./results/params'):
    output_path = Path(output_folder) / f'params_{str(epoch).zfill(3)}.data'
    
    params_bytes = flax.serialization.to_bytes(params)
    with open(output_path, 'wb') as f:
        f.write(params_bytes)
    
    return output_path

def get_opt(params, max_lr):
    sched = optax.warmup_cosine_decay_schedule(1e-5, max_lr, 200, 50000, 1e-5)
    opt = optax.adamw(sched, weight_decay=0.1)
    opt = optax.apply_if_finite(opt, 20)
    params =  optax.LookaheadParams(params, deepcopy(params))
    opt = optax.lookahead(opt, 5, 0.5)
    opt_state = opt.init(params)

    return opt, opt_state, params

def train(model, num_feat, log, tr_loader, valid_loader, cfg, params=None, max_lr=1e-4, num_epochs=100, min_pr=0.1):
    rng = random.PRNGKey(42**2)
    x = jnp.ones(( num_feat,), dtype=jnp.float32)

    params_init = model.init(random.PRNGKey(0), x, rng, train=False)

    if params is None:
        params = params_init

    opt, opt_state, params = get_opt(params, max_lr)

    @jax.jit
    def val_loss_func(params, x_b, y_b):
        # Negative Log Likelihood loss
        def nll(x, y):
            return -model.apply(params, x, y, False, method=model.log_prob)

        # Use vamp to vectorize over the batch dim.
        return jax.vmap(nll)(x_b, y_b).mean()

    @jax.jit
    def tr_loss_func(params, x_b, y_b, rng):
        # Negative Log Likelihood loss
        def loss(x, y, rng_drop):
            return  model.apply(params, x, y, True, method=model.log_prob, rngs = {'dropout': rng_drop},  )

        # Use vamp to vectorize over the batch dim.
        rngs = random.split(rng, num=len(x_b))
        log_p = jax.vmap(loss)(x_b, y_b, rngs)
        return (-log_p).mean()
    
    @jax.jit
    def step(x, y, params, opt_state, rng):
        loss, grads = jax.value_and_grad(tr_loss_func)(params.fast, x, y, rng)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return loss, params, opt_state, random.split(rng)[1]

    
    @jax.jit
    def val_step(x_b, y_b, params, rng):
        loss = val_loss_func(params.slow, x_b, y_b)

        def get_sample(x, rng):
            rng_drop, sample_rng = random.split(rng)
            return model.apply(params.slow, x, sample_rng, False, method=model.sample,
                               rngs={'dropout': rng_drop})

        rngs = random.split(rng, num=len(x_b))
        pred = jax.vmap(get_sample)(x_b, rngs)
        return loss, pred, random.split(rng)[1]

    best_loss = jnp.inf
    for epoch in range(1, num_epochs+1):
        tr_loss = []
        for n, batch in enumerate(tr_loader):
            #assert not np.isnan(batch[0]).any()
            loss, params, opt_state, rng = step(batch[0], batch[1], params, opt_state, rng)

            wandb.log({'train_loss_step' : loss})
            tr_loss.append(loss)
        
        param_path = save_params(params.slow, epoch, output_folder=cfg.param_path)

        # Calculate the validation loss and generate some samples.
        val_loss = []
        preds = []
        targets = []
        for batch in valid_loader:
            x, y = batch
            loss, pred, rng = val_step(x, y, params, rng)
            val_loss.append(loss)
            preds.append(pred)
            targets.append(y)

        preds = jnp.concatenate(preds, axis=None)
        targets = jnp.concatenate(targets, axis=None)

        print(f'preds std : {preds.std()}')
        print(f'target std : {targets.std()}')

        # Calculate the expected number of rain days
        dd_target = (targets < min_pr).sum()/targets.size
        dd_pred = (preds < min_pr).sum()/preds.size
        
        val_loss = jnp.stack(val_loss).mean()
        tr_loss = jnp.stack(tr_loss).mean()

        log({'train_loss_epoch': tr_loss,
             'val_loss': val_loss,
             'epoch': epoch,
             'dd_val' : dd_pred,
             'dd_mae' : jnp.abs(dd_target - dd_pred)})

        print(f'{epoch}/{num_epochs}, valid loss : {val_loss:.4f}, train loss'
              f' {tr_loss:.4f}, dry day prob {dd_pred:.4f}, expected {dd_target:.4f}')

        make_qq(preds, targets, epoch, log, output_folder=cfg.plot_path)


        # if epoch > 2 and val_loss < best_loss:
        #     best_loss = val_loss
        #     wandb.log_artifact( str(param_path), name='params_' + str(epoch).zfill(3), type='training_weights') 

    return params.slow

def load_params(model, path, num_feat): 
    x = jnp.zeros((num_feat,), dtype=jnp.float32)
    params = model.init(random.PRNGKey(0), x, random.PRNGKey(42))

    print(f'Loading params from {path}...')
    with open(path, 'rb') as f:
        params = flax.serialization.from_bytes(params, f.read())

    return params

def parse_model(model):
    if isinstance(model, str):
        return getattr(spg_dist, model)()
    if isinstance(model, list):
        return [parse_model(mod) for mod in model]
    if isinstance(model, dict):
       # Dictionaries should only have one item
       for k,v in model.items():
           return getattr(spg_dist, k)(parse_model(v))
    else:
        raise ValueError(f'Invalid model {model}') 
    

def get_model(version=None, path=None):
    assert not (path is None and version is None), 'You need to pass a path or a version of the model.'
    
    if path is None:
        path = Path(__file__).parent.parent / 'config' / 'models' / f'{version}.yml'

    with open(path, 'r') as f:
        model_dict = yaml.load(f, Loader=yaml.FullLoader)

    print(f'Model: {model_dict}')
        
    dist = parse_model(model_dict['model'])
    base_cls = getattr(spg_dist, model_dict['base_spg'])
    return base_cls(dist), model_dict


def get_config(name, version, location, ens=None, output_path=None):
    with open(Path('config/') / (name + '.yml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = ConfigDict(cfg, type_safe=False)
    
    if ens is not None:
        cfg.ens_path = Path(cfg.ens_path) / version / f'{location}_epoch_{str(ens).zfill(3)}'
        cfg.ens_path.mkdir(parents=True, exist_ok=True)

    # Optionally overwite output path
    if output_path is not None:
        cfg.output_path = output_path 
        
    cfg.output_path = Path(cfg.output_path) / version / location 

    cfg.output_path.mkdir(parents=True, exist_ok=True)
    cfg.stats_path = cfg.output_path / 'stats.json'
    
    cfg.param_path = cfg.output_path / 'params'
    cfg.param_path.mkdir(parents=True, exist_ok=True)
    cfg.plot_path = cfg.output_path / 'plots'
    cfg.plot_path.mkdir(parents=True, exist_ok=True)

    
    cfg.input_path = Path(cfg.input_path)
    
    # if 'base_freq' in cfg:
    #     cfg.input_file = cfg.input_path / (location + f'_{cfg.base_freq}.nc')
    # else:
    cfg.input_file = cfg.input_path / (location + '.nc')

    cfg.location = location
    cfg.version = version

    return cfg

def train_wh(cfg, location, bs=256, load_stats=False, **kwargs):
    print('Loading weather@home')
    wh_ens = data_utils.load_wh(location=location, num_ens=500)
    print(f'Loaded {len(wh_ens)} ens')

    ds_train, ds_valid = data_loader.get_datasets(wh_ens, num_valid=50, is_wh=True, ds_cls=data_loader.PrecipitationDatasetWH,
                                                  stats_path=cfg.stats_path, load_stats=load_stats)

    tr_loader, val_loader = data_loader.get_data_loaders(ds_train, ds_valid, bs=bs)
    num_feat = len(ds_train[0][0])
    print(f'Num features {num_feat}')

    train(cfg=cfg, num_feat=num_feat, tr_loader=tr_loader, valid_loader=val_loader, **kwargs)


def train_obs(model, cfg, load_stats=False, params_path=None, freq='H', resample=32, bs=256, param_init=None, **kwargs):
    data = data_utils.load_nc(cfg.input_file)

    if resample is not None:
        assert freq == 'H'
        freq = f'{resample}H'
        data = data_utils.average(data, resample)

    ds_train, ds_valid = data_loader.get_datasets(data, num_valid=cfg.num_valid, load_stats=load_stats,
                                                  stats_path=cfg.stats_path, freq=freq, ds_cls=data_loader.PrecipitationDataset)

    tr_loader, val_loader = data_loader.get_data_loaders(ds_train, ds_valid, bs=bs)
    
    num_feat = len(ds_train[0][0])
    print(f'Num features {num_feat}')
    
    if params_path is not None:
        params = load_params(model, params_path, num_feat)
    else:
        params = param_init

    train(model, num_feat=num_feat, tr_loader=tr_loader, valid_loader=val_loader, params=params, cfg=cfg, **kwargs)


def train_multiscale(model, cfg, load_stats=False, params_path=None, bs=256, **kwargs):
    data = data_utils.load_nc(cfg.input_file)

    ds_train, ds_valid = data_loader.get_datasets(data, num_valid=365*3*24, load_stats=load_stats, 
                                                  stats_path=cfg.stats_path, ds_cls=data_loader.PrecipitationDatasetMultiScale,
                                                  avg_period=[1, 2, 4, 8, 16, 32])

    tr_loader, val_loader = data_loader.get_data_loaders(ds_train, ds_valid, bs=bs)
    num_feat = len(ds_train[0][0])
    print(f'Num features {num_feat}')

    if params_path is not None:
        params = load_params(model, params_path, num_feat)
    else:
        params = None
    
    train(model, num_feat=num_feat, tr_loader=tr_loader, valid_loader=val_loader, params=params, cfg=cfg, **kwargs)

def wh_fine_tune_obs(loc, version, load_stats=False, bs=256, wh_epochs=20, train_split=False,
                     output_path='/mnt/temp/projects/otago_uni_marsden/data_keep/spg/training_params/split/'):
    # Train with w@h first then fine tune using obs.
    
    # Train the hourly splitter
    if train_split:
        print('Training hourly splitter -----')
        version_split = version + '_split'
        cfg_split = get_config('base_32H', version=version_split, location=loc, output_path=output_path)
        model, model_dict = get_model(version_split)

        wandb.init(entity='bodekerscientific', project='SPG', config={'cfg' : cfg_split, 'model_dict' : model})
        logger = wandb.log
        
        train_multiscale(model, cfg_split, load_stats=load_stats,  bs=bs, log=logger, num_epochs=20, max_lr=1e-3)
        wandb.finish()

    # Load the 32H stats config
    version_32H = version + '_32H'
    cfg_32H = get_config('base_32H', version=version_32H, location=loc, output_path=output_path)

    # Train wh
    print('Training w@h -----')
    version_wh = version + '_wh'
    cfg_wh = get_config('base_32H', version=version_wh, location=loc, output_path=output_path)
    model, model_dict = get_model(version + '_daily')

    # Overwrite the stats path so we can resuse when we train the 32H version
    cfg_wh.stats_path = cfg_32H.stats_path

    wandb.init(entity='bodekerscientific', project='SPG', config={'cfg' : cfg_wh, 'model_dict' : model})
    logger = wandb.log
    
    param_wh = train_wh(cfg_wh, loc.split('_')[0], model=model, load_stats=False,  bs=bs, num_epochs=wh_epochs,
                        log=logger, max_lr=1e-3)
    wandb.finish()

    # Fine tune with obs
    print('Fine tuning with obs')
    wandb.init(entity='bodekerscientific', project='SPG', config={'cfg' : cfg_32H, 'model_dict' : model})
    logger = wandb.log
    
    train_obs(model, cfg_32H, load_stats=True,  bs=bs, log=logger, max_lr=1e-4, num_epochs=20, param_init=param_wh)
    wandb.finish()

if __name__ == '__main__':
    if len(sys.argv) == 3:
        location = sys.argv[1]
        version = sys.argv[2]
    else:
        raise ValueError('You need to pass the run location, version and epoch number' \
                          ' as an argument, e.g python spg/train_spg.py dunedin v7')
    wh_fine_tune_obs(location, version=version)
    
    #bs = 256
    #version_split = version + '_split'
    #cfg = get_config('base_32H', version=version_split, location=loc, output_path=output_path)
    #model, model_dict = get_model(version_split)
    #wandb.init(entity='bodekerscientific', project='SPG', config={'cfg' : cfg, 'model_dict' : model})
    #logger = wandb.log
    

    #'/mnt/temp/projects/otago_uni_marsden/data_keep/spg/training_params/hourly/v7/dunedin/params/params_022.data'
    #train_obs(model=model, log=logger, cfg=cfg, bs=bs, resample=32, load_stats=True)
    
    #train_daily(model=model, log=wandb.log, params_path=params_path, )
    #train_wh(model=model, log
    # =wandb.log)
    
    # rng = random.PRNGKey(42)
    # y = jnp.array([1.0, 2.0, 0, 0.04, 1.0, 1.0]).astype(jnp.float32)
    # x = y[:, None]
    
    # variables = model.init(random.PRNGKey(0), x, rng)    
    # probs = model.apply(variables, x, y, method=model.log_prob)
