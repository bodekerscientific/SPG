"""
    Evaluates the SPG and generates ensambles

    @Author Leroy Bird
    
    You might need to set, as TF and JAX will both try grab the full gpu memory.
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
"""
from collections import defaultdict
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
from datetime import datetime
from bslibs.plot.qqplot import qqplot

import pandas as pd
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import get_context

from spg.generator import SPG
from spg import distributions, jax_utils, data_utils, generator, data_loader, train_spg
from pathlib import Path
from itertools import product
import numpy as np
from jax.experimental.host_callback import id_print

import flax
from tqdm import tqdm
import jax

N_CPU = 24

def cond_func(values, last_cond):
    rain = values[-1]
    is_rain = float(rain > 0)

    return {k: generator.cycle(last_cond[k], v) if last_cond[k] is not None else None
            for k, v in zip(['rainday', 'rain'], [is_rain, rain])}


def cond_func_ns(values, last_cond, data=None):
    rain = values[-1]
    is_rain = float(rain > 0)

    # The index will be the length of values, used to get t_prime.
    idx = len(values) - 1

    cond_next = {'rainday': generator.cycle(last_cond['rainday'], is_rain)}
    cond_next['rain'] = {'tprime': data['tprime'][idx]}
    return cond_next


def plot_qq(target, predictions, output_path):
    plt.figure(figsize=(12, 8))
    qqplot(target, predictions, linewidth=0)
    plt.xlabel('Observations [mm/day]')
    plt.ylabel('Predictions [mm/day]')
    max_val = max(target.max(), predictions.max())
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.savefig(output_path,  dpi=250)
    plt.close()


def gen_preds(sp: SPG, data: pd.Series, start_date='1950-1-1', end_date='2100-1-1', 
              plot_path=Path('./qq.png'), tprime=None, freq='D'):

    times = pd.date_range(start=start_date, end=end_date, freq=freq)
    rd = sp.rainday
    cond = {'rain': None, 'rainday': jnp.array([[1]*rd.ar_depth])}

    if tprime is not None:
        tprime = data_utils.get_tprime_for_times(times, tprime)
        cond['rain'] = {'tprime' : tprime[0]}
        predictions = sp.generate(num_steps=len(times), cond_init=cond, 
                                  cond_func=partial(cond_func_ns, data={'tprime' : tprime}))
    else:
         predictions = sp.generate(num_steps=len(times), cond_init=cond, 
                                  cond_func=cond_func)


    print(f'{(predictions >= rd.thresh).sum()/predictions.size} expected {(data >= rd.thresh).sum()/data.size}')

    if plot_path:
        plot_qq(data, predictions, output_path=plot_path)

    return pd.Series(predictions, index=times)

 
def setup_output(data, feat_samples, start_date, end_date, spin_up_steps=100, freq='H'):
    pr_re = data.resample(freq, origin='start').asfreq()
    
    if len(freq) > 1:
        assert 'H' in freq
        count = int(''.join([d for d in freq if d.isdigit()]))
    else:
        count = 1
        
    
    dts = pd.date_range(start=start_date - pd.Timedelta(spin_up_steps*count, unit='H',), end=end_date, freq=freq)
    
    # Find the first index where we get non nan values
    idx_valid = np.where(~np.isnan(pr_re.rolling(feat_samples+1).mean().values))[0][0]
    output = pd.Series(np.zeros(len(dts), dtype=np.float32), index=dts)
    output.index.name = 'time'

    # Copy of the initial values from the obs
    output.iloc[0:feat_samples] = pr_re[idx_valid-feat_samples+1:idx_valid+1].values 
    return output

def run_spg_mlp(data : pd.Series, params_path : Path, stats : dict, cfg, feat_samples = 6*24, spin_up_steps=1000, 
                start_date=None, end_date=None, rng=None, sce='ssp245', use_tqdm=True, freq='H', max_pr=100):

    #assert freq in ['H', 'D'], 'Only hourly and daily SPGs are supported at this time.'
    model, model_dict = train_spg.get_model(cfg.version)

    
    feat_func =  data_loader.generate_features if freq == 'H' else data_loader.generate_features_daily
    feat_func = partial(feat_func, **model_dict['loader_args'])
    
    feat_func_norm = lambda x : data_loader.apply_stats(stats['x'], feat_func(x, sce=sce)[0],)

    if rng is None:
        rng = random.PRNGKey(np.random.randint(1e10))
    if start_date is None:
        start_date = data.index.min()
    if end_date is None:
        end_date = data.index.max()
    
    output = setup_output(data, feat_samples, start_date, end_date, spin_up_steps=spin_up_steps, freq=freq)

    num_feat = feat_func_norm(output.iloc[0:feat_samples+1]).shape[1]
    params = train_spg.load_params(model, params_path, num_feat)
    
    @jax.jit
    def sample_func(x, rng):
        return model.apply(params, x, rng, method=model.sample)

    n_range = range(feat_samples, len(output))
    if use_tqdm:
        n_range = tqdm(n_range)

    for n in n_range:
        subset_cond = output.iloc[n - feat_samples:n+1]
        x = feat_func_norm(subset_cond)
        
        # Sometimes (very rarely) we get an np.nan
        # So just repeat until we get a valid output
        out = jnp.nan
        while(not jnp.isfinite(out) or out > max_pr):
            rng, sample_rng = random.split(rng)
            out = sample_func(x[0], sample_rng)
            if not jnp.isfinite(out):
                print('Got invalid value!!')
            else:
                # Denormalise
                out  = data_loader.inverse_stats(stats['y'], out)
                if out > max_pr:
                    print(f'Got a 1{freq} precipitation of {out}')

        output.values[n] =  out

    # Remove the real data and spin up period from the output
    output = output.iloc[spin_up_steps:]

    return output
    

def run_save_proj(ens_args, cfg, data : pd.Series, params_path : Path, 
                  start_date=pd.Timestamp(1980, 1, 1), end_date=pd.Timestamp(2100, 1, 1), freq='H', **kwargs):

    sce, ens_num, idx = ens_args
    rng = random.PRNGKey(seed=971*(int(idx)+42))
    preds = run_spg_mlp(data, params_path, start_date=start_date, end_date=end_date, rng=rng, freq=freq, cfg=cfg, **kwargs)

    output_path = cfg.ens_path / f'{cfg.location}_{sce}_{str(ens_num).zfill(3)}.nc'    
    data_utils.save_nc_tprime(preds, output_path, sce=sce, units='mm/hr' if freq == 'H' else f'mm/{freq}')


def run_pool(cfg, **kwargs):
    ens = list(product(cfg.scenarios, np.arange(cfg.num_ens) + 1))
    idxs = np.arange(len(ens)) 
    ens = np.concatenate([np.array(ens), idxs[:, None]], axis=1)

    print(f'Running {len(ens)} scenarios')

    with get_context('spawn').Pool(N_CPU) as p:
        p.map(partial(run_save_proj, use_tqdm=True, cfg=cfg, **kwargs), ens)

def get_params_path(cfg, epoch):
    return cfg.param_path / f'params_{str(epoch).zfill(3)}.data'


def run_spg_multi(data_daily : pd.Series, params_path : Path, stats : dict, cfg, rng=None, x_cond=4, 
                  output_freq='H', in_freq = 24, use_tqdm=True, max_pr=100):
    
    
    data_daily.values[data_daily.values < 1e-6] = 0.0

    def get_x(last_pr, total_pr, freq):
        return jnp.concatenate([jnp.array(last_pr[-x_cond:]), total_pr, freq], axis=None).astype(jnp.float32)

    get_x_norm = lambda *args : data_loader.apply_stats(stats['x'], get_x(*args))

    if rng is None:
        rng = random.PRNGKey(np.random.randint(1e10))
        
    # We assume there are no gaps in the input
    new_idx = pd.date_range(start=data_daily.index.min(), periods=len(data_daily)*in_freq, freq=output_freq)
    output = pd.Series(np.zeros(len(new_idx)), index=new_idx)

    num_feat = len(get_x_norm([0.0]*x_cond, 0.0, 2))
        
    model, model_dict = train_spg.get_model(cfg.version)
    print(model_dict)
    params = train_spg.load_params(model, params_path, num_feat)
    
    @jax.jit
    def sample_func(x, rng):
        return model.apply(params, x, rng, method=model.sample)

    n_range = range(len(data_daily))
    if use_tqdm:
        n_range = tqdm(n_range)

    @jax.jit
    def split_pr(pr_last, pr, freqs, rng):
        pr_last = list(pr_last)
        for freq in freqs:
            x  = get_x_norm(last_pr, pr, freq)

            rng, sample_rng = random.split(rng)
            ratio = sample_func(x, sample_rng)
            pr_l = (1.0 - ratio)*pr
            # The remaining precipitation in used as the new total precipitation
            pr = pr*ratio
            pr_last.append(pr_l)
            
        pr_last.append(pr)    
        
        assert len(pr_last) == in_freq*2
        return pr_last[in_freq:], rng
    
    all_freq = jnp.array(list(reversed(range(2, in_freq+1))))
    
    # We assume it wasn't raining over the first day
    last_pr = [0.0]*in_freq
    output_lst = []
    for n in n_range:
        daily_pr = data_daily.values[n]
        if daily_pr > 0:
            last_pr, rng = split_pr(last_pr, daily_pr , all_freq, rng)
        else:
            last_pr = [0.0]*in_freq
            
        output_lst.append(last_pr)
        
    output_lst = np.concatenate(output_lst, axis=None)
    output[:] = output_lst
    
    return output

def run_hourly():
    if len(sys.argv) == 4:
        location = sys.argv[1]
        version = sys.argv[2]
        param_epoch = int(sys.argv[3])
    else:
        raise ValueError('You need to pass the run location, version and epoch number' \
                          ' as an argument, e.g python spg/run.py dunedin v7 12')


    print(f'Training {version} for {location} using epoch {param_epoch}')

    cfg = train_spg.get_config('base_hourly', version, location, param_epoch)
    data = data_utils.load_nc(cfg.input_file)
    print(cfg)
    max_pr = cfg['max_values'][location]

    stats = data_loader.open_stats(cfg.stats_path)
    param_path = get_params_path(cfg, param_epoch)

    preds = run_spg_mlp(data, param_path, stats=stats, cfg=cfg, freq='H', max_pr=max_pr)
    data_utils.save_nc_tprime(preds, cfg.ens_path / (location + '.nc'))

    run_pool(cfg=cfg, data=data, stats=stats, params_path=param_path, freq='H', max_pr=max_pr)

def run_save_split(ens_args, cfg, params_path : Path, stats, **kwargs):
    fname, idx = ens_args
    print(f'Processing {fname}')
    
    data = data_utils.load_nc(fname)
    rng = random.PRNGKey(seed=901*(int(idx)+42))
    fname_split = fname.name.split('_')
    
    if len(fname_split) >= 2:
        sce = fname_split[-2]
    else:
        print(f"Using default scenario")
        sce = 'rcp26'
    if sce not in cfg.scenarios:
        print(f"Can't find {sce} in scenarios")
        sce = 'rcp26'
    
    preds = run_spg_multi(data, params_path, stats=stats, cfg=cfg, rng=rng, **kwargs)
    
    output_path = cfg.ens_path /  fname.name
    print(f'Saving to {output_path}, {sce}')
    data_utils.save_nc_tprime(preds, output_path, units='mm/hr', sce=sce)

def run_multiscale():
    if len(sys.argv) == 5:
        location = sys.argv[1]
        version = sys.argv[2]
        param_epoch_split = int(sys.argv[3])
        param_epoch_32H = int(sys.argv[4])
    else:
        raise ValueError('You need to pass the run location, version and epoch number for the splitter then the hourly dataset' \
                          ' as an argument, e.g python spg/run.py dunedin v7 12 2')


    print(f'Training {version} for {location} using epoch {param_epoch_split} for split and {param_epoch_32H} for 24HR of precip')

    # Run the 32 hour spg.
    #cfg = train_spg.get_config('base_32H', version +'_32H', location, ens=param_epoch_32H)
    # data = data_utils.load_nc(cfg.input_file)
    
    # max_pr = cfg['max_values'][location]
    # stats = data_loader.open_stats(cfg.stats_path)
    
    # param_path = get_params_path(cfg, param_epoch_32H)
    # preds = run_spg_mlp(data, param_path, stats=stats, cfg=cfg, freq='32H', max_pr=max_pr)
    
    # data_utils.save_nc_tprime(preds, cfg.ens_path / (location + '.nc'), units='mm/32H')
    # run_pool(cfg=cfg, data=data, stats=stats, params_path=param_path, freq='32H', max_pr=max_pr)

    # Run the splitter
    cfg_split = train_spg.get_config('base_32H', version +'_split', location, ens=param_epoch_split)
    stats = data_loader.open_stats(cfg_split.stats_path)
    param_path = get_params_path(cfg_split, param_epoch_split)
        
    #ens = list(Path(cfg.ens_path).glob(f'{location}.nc')) 
    ens = [Path(f'/mnt/temp/projects/otago_uni_marsden/data_keep/spg/station_data_hourly/{location}_daily.nc')]
    idxs = np.arange(len(ens))
    ens = np.stack([np.array(ens), idxs], axis=1)

    print(f'Running {len(ens)} scenarios')

    with get_context('spawn').Pool(N_CPU) as p:
        p.map(partial(run_save_split, stats=stats, params_path=param_path, use_tqdm=True, cfg=cfg_split,), ens)

# def run_daily():
#     # TODO: Update with config file
#     fname_obs = 'dunedin'
#     version = 'v3'
#     output_path = Path('/mnt/temp/projects/otago_uni_marsden/data_keep/spg/')
#     save_obs = True
#     feat_samples = 8

#     output_path_obs = output_path / 'station_data' 
#     output_path_ens = output_path / 'ensemble' / version
#     output_path_ens.mkdir(exist_ok=True, parents=True)

#     data = data_utils.load_data()#'/mnt/datasets/NationalClimateDatabase/NetCDFFilesByVariableAndSite/Hourly/Precipitation/1962.nc')
#     param_path = f'params_daily_{version}.data'
#     stats = data_loader.open_stats('./stats.json')
    
#     run_kwargs = dict(data=data, stats=stats, params_path=param_path, feat_samples=feat_samples, freq='D'   )
    
#     preds = run_spg_mlp(**run_kwargs)
#     data_utils.save_nc_tprime(preds, output_path_ens / (fname_obs + '.nc'), units='mm/day')
    
#     run_pool(output_folder=output_path_ens, file_pre=fname_obs, **run_kwargs)

if __name__ == '__main__':
    run_multiscale()