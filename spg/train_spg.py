"""
    SPG using a single nn-module and a MLP for the distribution params
    
    @Author Leroy Bird
    
    You might need to set, as TF and JAX will both try grab the full gpu memory.
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
"""
from dataclasses import replace
import fire

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
from tqdm import tqdm

# from jax.config import config
# config.update('jax_disable_jit', True)


#Global flag to set a specific platform, must be used at startup.
#jax.config.update('jax_platform_name', 'cpu')

def make_qq(preds: list, targets: list, averages=[1, 8, 24], title=None):
    # Ensure output folder exists
    _, axes = plt.subplots(1, len(averages), figsize=(len(averages)*8, 8))
    if title:
        plt.suptitle(title)
        
    for ax, av in zip(axes, averages):
        # calculate a n-day average.
        def rolling(arr):
            filt = np.ones(av, dtype=np.float32)
            return np.convolve(arr, filt, mode='valid')
        ax.set_title(f'{av} average')
        qqplot(rolling(targets), rolling(preds), ax=ax, num_highlighted_quantiles=10, linewidth=0)
        
def save_plot(name, epoch: int, log=None, output_folder='./results/plots'):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f'{name}_{str(epoch).zfill(3)}.png'
    plt.savefig(output_path, transparent=False, dpi=150)
    plt.close()
    
    if log is not None:
        log({'qq_plot' : wandb.Image(str(output_path))})

    
def save_params(params, epoch : int, output_folder='./results/params'):
    # Ensure output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    output_path = output_folder / f'params_{str(epoch).zfill(3)}.data'
    
    params_bytes = flax.serialization.to_bytes(params)
    with open(output_path, 'wb') as f:
        f.write(params_bytes)
    
    return output_path

def get_opt(params, max_lr=1e-3, min_lr=1e-8, max_steps=50000, spin_up_steps=300, wd=0.01, lookahead=5):
    sched = optax.warmup_cosine_decay_schedule(min_lr, max_lr, spin_up_steps, max_steps, min_lr)
    opt = optax.adamw(sched, weight_decay=wd)
    opt = optax.apply_if_finite(opt, 20)
    params =  optax.LookaheadParams(params, deepcopy(params))
    opt = optax.lookahead(opt, lookahead, 0.5)
    opt_state = opt.init(params)

    return opt, opt_state, params

def set_tprime_batch(x, stats, t_prime):
    # It is assumed that t_prime is the last variable in x
    x_stats = stats['x']
    return x.at[:, -1].set( (t_prime - x_stats['mean'][-1])/ x_stats['std'][-1])
    
def set_tprime(x, stats, t_prime):
    # It is assumed that t_prime is the last variable in x
    x_stats = stats['x']
    return x.at[-1].set( (t_prime - x_stats['mean'][-1])/ x_stats['std'][-1])
    

def train(model, num_feat, tr_loader, valid_loader, log=None, cfg=None, params=None, num_epochs=40, opt_kwargs={},
          make_tp_plots=False, use_lin_loss=True):
    
    rng = random.PRNGKey(42**2)
    x = jnp.ones(( num_feat,), dtype=jnp.float32)
    
    if params is None:
        params = model.init(random.PRNGKey(0), x, rng, train=False)

    opt, opt_state, params = get_opt(params, **opt_kwargs)

    @jax.jit
    def val_loss_func(params, x_b, y_b):
        # Negative Log Likelihood loss
        def nll(x, y):
            return -model.apply(params, x, y, False, method=model.log_prob)
        
        # Use vamp to vectorize over the batch dim.
        return jax.vmap(nll)(x_b, y_b).mean()

    @jax.jit
    def tr_nll(params, x_b, y_b, rng):
        # Negative Log Likelihood loss
        def loss(x, y, rng_drop):
            return model.apply(params, x, y, True, method=model.log_prob, rngs = {'dropout': rng_drop})
            
        # Use vamp to vectorize over the batch dim.
        rngs = random.split(rng, num=len(x_b))
        log_p = jax.vmap(loss)(x_b, y_b, rngs)
        return (-log_p).mean()
    
    
    
    @jax.jit
    def tr_lin_loss_func(params, x_b, y_b, rng):
        loss = tr_nll(params, x_b, y_b, rng)
        
        def loss_lin(x, rng, eps=1e-5):    
            t_p_1, t_p_2, t_p_3, p_dist = random.uniform(rng, (4,))
            
            # Generate some random samples for t_prime
            t_p_1 *= 1.5 
            t_p_2 = t_p_2*2 + 0.25 + t_p_1
            t_p_3 = 2*t_p_3 + t_p_2 + 0.25
            
            x_pred_1 = model.apply(params, set_tprime(x, valid_loader.dataset.stats, t_p_1), p_dist, False, method=model.ppf_wet, rngs = {'dropout': rng})
            x_pred_2 = model.apply(params, set_tprime(x, valid_loader.dataset.stats, t_p_2), p_dist, False, method=model.ppf_wet, rngs = {'dropout': rng}) 
            x_pred_3 = model.apply(params, set_tprime(x, valid_loader.dataset.stats, t_p_3), p_dist, False, method=model.ppf_wet, rngs = {'dropout': rng})
            
            def loss(): 
                r1 = jnp.log((eps + x_pred_2)/(x_pred_1 + eps))/(t_p_2 - t_p_1)
                r2 = jnp.log((eps + x_pred_3)/(x_pred_1 + eps))/(t_p_3 - t_p_1)                          
                return jnp.abs(r1 - r2)
             
                #return jnp.abs((x_pred_2 - x_pred_1)/(t_p_2 - t_p_1)  - (x_pred_3 - x_pred_1)/(t_p_3 - t_p_1))
            
            # For numerical stability at the start of training
            cond = (x_pred_1 < 200) & (x_pred_2 < 200) & (x_pred_3 < 200)
            
            return jax.lax.cond(cond, loss, lambda :  0.0)
        
        if use_lin_loss:
            rngs = random.split(rng, num=len(x_b))
            loss += jax.vmap(loss_lin)(x_b, rngs).mean() 
            
        return loss
        
    
    @jax.jit
    def step(x, y, params, opt_state, rng):
        rng, rng_fit = random.split(rng)
        loss, grads = jax.value_and_grad(tr_lin_loss_func)(params.fast, x, y, rng_fit)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return loss, params, opt_state, rng
    
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

    def valid_loop(rng, replace_tprime = None):
        val_loss = []
        preds = []
        targets = []
        for batch in valid_loader:
            x, y = batch
            if replace_tprime is not None:
                #x.at
                x = set_tprime_batch(x, valid_loader.dataset.stats, replace_tprime)
                
            loss, pred, rng = val_step(x, y, params, rng)
            val_loss.append(loss)
            preds.append(pred)
            targets.append(y)

        preds = jnp.concatenate(preds, axis=None)
        targets = jnp.concatenate(targets, axis=None)
    
        return preds, targets, val_loss, rng
    
    best_loss = jnp.inf
    for epoch in range(1, num_epochs+1):
        tr_loss = []
        tr_it = tqdm(tr_loader)
        for batch in tr_it:
            #assert not np.isnan(batch[0]).any()
            loss, params, opt_state, rng = step(batch[0], batch[1], params, opt_state, rng)
            if log is not None:
                wandb.log({'train_loss_step' : loss})
            tr_loss.append(loss)
            tr_it.set_description(f'Train loss : {jnp.stack(tr_loss[-50:]).mean():.3f}')
        
        if cfg is not None:
            param_path = save_params(params.slow, epoch, output_folder=cfg.param_path)

        preds, targets, val_loss, rng = valid_loop(rng)
        preds = preds.at[preds <=  model.min_pr].set(0.0)

        print(f'preds std : {preds.std()}')
        print(f'target std : {targets.std()}')
        

        # Calculate the expected number of rain days
        dd_target = (targets <=  model.min_pr).sum()/targets.size
        dd_pred = (preds <=  model.min_pr).sum()/preds.size
        
        tr_loss = jnp.stack(tr_loss).mean()
        val_loss = jnp.stack(val_loss).mean()

        rms_sorted_loss = jnp.mean((jnp.sort(targets) - jnp.sort(preds))**2)**0.5
        
        if log is not None:    
            log({'train_loss_epoch': tr_loss,
                'val_loss': val_loss,
                'epoch': epoch,
                'dd_val' : dd_pred,
                'dd_mae' : jnp.abs(dd_target - dd_pred),
                'rms_loss' : rms_sorted_loss})

        print(f'{epoch}/{num_epochs}, valid loss : {val_loss:.4f}, train loss  {tr_loss:.4f}, '
              f'rms loss {rms_sorted_loss:.4f}, dry day prob {dd_pred:.4f}, expected {dd_target:.4f}')
        
        if cfg is not None:
            plot_folder = cfg.plot_path
        else:
            plot_folder = './results/plots'
        
        title = f'Validation loss: {val_loss:.4f}, rms loss {rms_sorted_loss:.4f} '
        make_qq(preds[0:20000], targets[0:20000], title=title)
        save_plot('qq', epoch, log, output_folder = plot_folder)
        
        if make_tp_plots:
            val_tprime = np.arange(0, 2.5, 0.2)
            
            losses_tprime = []
            for t_prime in val_tprime:
                _, _, val_loss_t, rng = valid_loop(rng, replace_tprime=t_prime)
                losses_tprime.append(jnp.stack(val_loss_t).mean())
            
            plt.plot(val_tprime, losses_tprime)
            save_plot('tprime_loss', epoch, log, output_folder = plot_folder)
            
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
    

def get_model(version=None, path=None, stats=None):
    assert not (path is None and version is None), 'You need to pass a path or a version of the model.'
    
    if path is None:
        path = Path(__file__).parent.parent / 'config' / 'models' / f'{version}.yml'

    with open(path, 'r') as f:
        model_dict = yaml.load(f, Loader=yaml.FullLoader)

    print(f'Model: {model_dict}')
        
    dist = parse_model(model_dict['model'])
    base_cls = getattr(spg_dist, model_dict['base_spg'])
    model = base_cls(dist)
    
    if stats is not None:
        model.min_pr /= stats['y']['std']
        print(model.min_pr)
        
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

def train_wh(cfg, location, bs=256, load_stats=False, data_obs=None, freq='D', **kwargs):
    print('Loading weather@home')
    _, model_dict = get_model(cfg.version)

    loader_args = model_dict['loader_args'] if 'loader_args' in model_dict else {}

    wh_ens = data_utils.load_wh(location=location, num_ens=500)
    print(f'Loaded {len(wh_ens)} ens')

    ds_train, ds_valid = data_loader.get_datasets(wh_ens, num_valid=50, is_wh=True, ds_cls=data_loader.PrecipitationDatasetWH,
                                                  stats_path=cfg.stats_path, load_stats=load_stats, **loader_args)

    tr_loader, val_loader = data_loader.get_data_loaders(ds_train, ds_valid, bs=bs)
    
    model, model_dict = get_model(cfg.version, stats=val_loader.dataset.stats)

    num_feat = len(ds_train[0][0])
    print(f'Num features {num_feat}')

    return train(model, cfg=cfg, num_feat=num_feat, tr_loader=tr_loader, valid_loader=val_loader, **kwargs)


def train_obs(cfg, model=None, load_stats=False, params_path=None, freq='H', bs=256, param_init=None, **kwargs):
    _, model_dict = get_model(cfg.version)

    data = data_utils.load_nc(cfg.input_file)
    
    loader_args = model_dict['loader_args'] if 'loader_args' in model_dict else {}

    ds_train, ds_valid = data_loader.get_datasets(data, num_valid=cfg.num_valid, load_stats=load_stats, stats_path=cfg.stats_path, 
                                                  freq=freq, ds_cls=data_loader.PrecipitationDataset, **loader_args)

    tr_loader, val_loader = data_loader.get_data_loaders(ds_train, ds_valid, bs=bs)
    
    model, model_dict = get_model(cfg.version, stats=val_loader.dataset.stats)
    
    num_feat = len(ds_train[0][0])
    print(f'Num features {num_feat}')
    
    if params_path is not None:
        params = load_params(model, params_path, num_feat)
    else:
        params = param_init

    return train(model, num_feat=num_feat, tr_loader=tr_loader, valid_loader=val_loader, params=params, cfg=cfg, **kwargs)


def train_multiscale(model, cfg, load_stats=False, params_path=None, bs=256, **kwargs):
    data = data_utils.load_nc(cfg.input_file)

    ds_train, ds_valid = data_loader.get_datasets(data, num_valid=365*3*24, load_stats=load_stats, 
                                                  stats_path=cfg.stats_path, ds_cls=data_loader.PrecipitationDatasetMultiScale)

    tr_loader, val_loader = data_loader.get_data_loaders(ds_train, ds_valid, bs=bs)
    num_feat = len(ds_train[0][0])
    print(f'Num features {num_feat}')

    if params_path is not None:
        params = load_params(model, params_path, num_feat)
    else:
        params = None
    
    train(model, num_feat=num_feat, tr_loader=tr_loader, valid_loader=val_loader, params=params, cfg=cfg, **kwargs)

def train_splitter(loc, version, load_stats=False, bs=256, wh_epochs=20, train_split=True,
                     output_path='/mnt/temp/projects/otago_uni_marsden/data_keep/spg/training_params/split/'):
    # Train with w@h first then fine tune using obs.
    print('Training hourly splitter -----')
    version_split = version + '_split'
    cfg_split = get_config('base_daily', version=version_split, location=loc, output_path=output_path)
    model, model_dict = get_model(version_split)

    wandb.init(entity='bodekerscientific', project='SPG', config={'cfg' : cfg_split, 'model_dict' : model})
    logger = wandb.log
    
    train_multiscale(model, cfg_split, load_stats=load_stats,  bs=bs, log=logger, num_epochs=20)
    wandb.finish()

def train_fine_tune_wh(cfg, load_stats=False, params_path=None, bs=256, **kwargs):
    # Train using weather@home, then fine tune on obs.
    print('Training w@h -----')
    
    #wandb.log
    param_wh = train_wh(cfg, cfg.location, load_stats=False,  bs=bs, num_epochs=20, **kwargs)
    
    # Fine tune with obs
    print('Fine tuning with obs')
    
    # Reduce the learning rate for fine tuning
    if 'opt_kwargs' in kwargs:
        kwargs['opt_kwargs']['max_lr'] = 1e-4
    else:
        kwargs['opt_kwargs'] = {'max_lr' : 1e-4} 
    
    train_obs(cfg, load_stats=True,  bs=bs, num_epochs=5, param_init=param_wh, **kwargs)
    
    #wandb.finish()

def run(location, cfg_name, model_version):
    cfg = get_config(cfg_name, version=model_version, location=location)
    wandb.init(entity='bodekerscientific', project='SPG', config={'cfg' : cfg})

    train_func = globals()[cfg.train_func]
    train_func(cfg, log=wandb.log, **cfg.train_kwargs)

if __name__ == '__main__':
    fire.Fire(run)
    # if len(sys.argv) == 3:
    #     location = sys.argv[1]
    #     version = sys.argv[2]
    # else:
    #     raise ValueError('You need to pass the run location, version and epoch number' \
    #                       ' as an argument, e.g python spg/train_spg.py dunedin v7')
    # wh_fine_tune_obs(location, version=version)
    
    #bs = 256
    #version_split = version + '_split'
    
    # cfg = get_config('base_hourly', version=version, location=location)
    # model, model_dict = get_model(version)
    # wandb.init(entity='bodekerscientific', project='SPG', config={'cfg' : cfg, 'model_dict' : model})
    # logger = wandb.log
    
    # print(model_dict['loader_args'])
    # #'/mnt/temp/projects/otago_uni_marsden/data_keep/spg/training_params/hourly/v7/dunedin/params/params_022.data'
    # train_obs(model=model, log=logger, cfg=cfg, resample=None, load_stats=False, ds_kwargs = model_dict['loader_args'])
    
    #train_daily(model=model, log=wandb.log, params_path=params_path, )
    #train_wh(model=model, log
    # =wandb.log)
    
    # rng = random.PRNGKey(42)
    # y = jnp.array([1.0, 2.0, 0, 0.04, 1.0, 1.0]).astype(jnp.float32)
    # x = y[:, None]
    
    # variables = model.init(random.PRNGKey(0), x, rng)    
    # probs = model.apply(variables, x, y, method=model.log_prob)
