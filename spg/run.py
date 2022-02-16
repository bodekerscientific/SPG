"""
    Evaluates the SPG and generates ensambles

    @Author Leroy Bird
    
    You might need to set, as TF and JAX will both try grab the full gpu memory.
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
"""
from distutils.command.config import config
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

N_CPU = 12


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


def param_func_scale_only(params, cond=None):
    # Only non-stationary for the scale param, as the shape is unstable.
    
    if cond is not None and 'tprime' in cond and cond['tprime'].size > 1:
        params_out = jnp.repeat(params[0:1, None], len(cond['tprime']), axis=1)
    else:
        params_out = params[:, None]
        
    return jnp.concatenate([params_out, jax_utils.linear_exp_split(params[1:], cond)], axis=0)


def fit_spg(data, use_tf=True, ar_depth=12, thresh=0.1):
    rd = distributions.RainDay(thresh=thresh, ar_depth=ar_depth)

    if use_tf:
        rain_dists = {0: distributions.TFGammaMix(num_mix=3),
                      0.99: distributions.TFGeneralizedPareto()}
    else:
        rain_dists = {0: distributions.SSWeibull(),
                      0.99: distributions.SSGeneralizedPareto()}
    rng = random.PRNGKey(42)

    sp = SPG(rd, rain_dists, rng)
    sp.fit(data.values)
    sp.print_params()

    return sp

def fit_spg_ns(data, t_prime, ar_depth=2, thresh=0.1):
    rd = distributions.RainDay(thresh=thresh, ar_depth=ar_depth)

    rain_dists = {0: distributions.TFWeibull(param_init=[0.75, 1.0, 0.5],
                                             param_func=param_func_scale_only),
                  0.99: distributions.TFGeneralizedPareto(param_init=[-1.0, 1.1, 0.01, 0.01],
                                                          param_func=jax_utils.linear_exp_split)}

    rng = random.PRNGKey(42)
    sp = SPG(rd, rain_dists, rng)

    sp.fit(data.values, cond={'tprime': t_prime})
    sp.print_params()

    return sp

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

def gen_save(arg, data, sp, df_magic, output_path, **kwargs):
    sce, ens_num, idx = arg
    sp.rnd_key = random.PRNGKey(seed=971*(int(idx)+42))

    predictions = gen_preds(sp, data, tprime=df_magic[sce], **kwargs)
    data_utils.make_nc(predictions, output_path / f'dunedin_{sce}_{str(ens_num).zfill(3)}.nc', tprime=df_magic[sce])

def run_non_stationary(output_path, data, scenario=['rcp26','rcp45','rcp60','rcp85','ssp119','ssp126',
                                                    'ssp245','ssp370','ssp434','ssp460','ssp585'], num_ens=4,
                                                    **kwargs):
    df_magic = data_utils.load_magic()
    t_prime = data_utils.get_tprime_for_times(data.index, df_magic['ssp245'])
    sp = fit_spg_ns(data, t_prime)

    ens = list(product(scenario, np.arange(num_ens) + 1))
    idxs = np.arange(len(ens)) 
    ens = np.concatenate([np.array(ens), idxs[:, None]], axis=1)

    print(f'Running {len(ens)} scenarios')

    with get_context('spawn').Pool(N_CPU) as p:
        p.map(partial(gen_save, data=data, sp=sp, df_magic=df_magic, output_path=output_path, **kwargs), ens)

def test_fit(data, **kwargs):
    sp = fit_spg(data)
    preds = gen_preds(sp, data, start_date='1950-1-1', end_date='1980-1-1', **kwargs)
    print(preds)
 
def setup_output(data, feat_samples, start_date, end_date, spin_up_steps=1000, freq='H'):
    pr_re = data.resample(freq).asfreq()
    dts = pd.date_range(start=start_date - pd.Timedelta(spin_up_steps, unit=freq,), end=end_date, freq=freq)
    
    # Find the first index where we get non nan values
    idx_valid = np.where(~np.isnan(pr_re.rolling(feat_samples+1).mean().values))[0][0]
    output = pd.Series(np.zeros(len(dts), dtype=np.float32), index=dts)
    output.index.name = 'time'

    # Copy of the initial values from the obs
    output.iloc[0:feat_samples] = pr_re[idx_valid-feat_samples+1:idx_valid+1].values 
    return output

def run_spg_mlp(data : pd.Series, params_path : Path, stats : dict, cfg, feat_samples = 6*24, spin_up_steps=1000, 
                start_date=None, end_date=None, rng=None, sce='ssp245', use_tqdm=True, freq='H'):

    assert freq in ['H', 'D'], 'Only hourly and daily SPGs are supported at this time.'
    
    feat_func =  data_loader.generate_features if freq == 'H' else data_loader.generate_features_daily
    feat_func_norm = lambda x : data_loader.apply_stats(stats['x'], feat_func(x, sce=sce)[0])

    if rng is None:
        rng = random.PRNGKey(np.random.randint(1e10))
    if start_date is None:
        start_date = data.index.min()
    if end_date is None:
        end_date = data.index.max()
    
    output = setup_output(data, feat_samples, start_date, end_date, spin_up_steps=spin_up_steps, freq=freq)

    num_feat = feat_func_norm(output.iloc[0:feat_samples+1]).shape[1]
    model = train_spg.get_model(cfg.version)
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
        while(not jnp.isfinite(out)):
            rng, sample_rng = random.split(rng)
            out = sample_func(x[0], sample_rng)
            if jnp.isnan(out):
                print('Got invalid value!!')
            # if not jnp.isfinite(out):
            #     id_print(out)
        output.iloc[n] = out

    # Remove the real data and spin up period from the output
    output = output.iloc[spin_up_steps:]

    # Denormalise
    output.values[:] = data_loader.inverse_stats(stats['y'], output.values)

    return output
    

def run_save_proj(ens_args, cfg, data : pd.Series, params_path : Path, 
                  start_date=pd.Timestamp(1980, 1, 1), end_date=pd.Timestamp(2100, 1, 1), freq='H', **kwargs):

    sce, ens_num, idx = ens_args
    rng = random.PRNGKey(seed=971*(int(idx)+42))
    preds = run_spg_mlp(data, params_path, start_date=start_date, end_date=end_date, rng=rng, freq=freq, cfg=cfg, **kwargs)

    output_path = cfg.ens_path / f'{cfg.location}_{sce}_{str(ens_num).zfill(3)}.nc'    
    data_utils.save_nc_tprime(preds, output_path, sce=sce, units='mm/hr' if freq == 'H' else 'mm/day')


def run_pool(cfg, **kwargs):
    ens = list(product(cfg.scenarios, np.arange(cfg.num_ens) + 1))
    idxs = np.arange(len(ens)) 
    ens = np.concatenate([np.array(ens), idxs[:, None]], axis=1)

    print(f'Running {len(ens)} scenarios')

    with get_context('spawn').Pool(N_CPU) as p:
        p.map(partial(run_save_proj, use_tqdm=True, cfg=cfg, **kwargs), ens)

def get_params_path(cfg, epoch):
    return cfg.param_path / f'params_{str(epoch).zfill(3)}.data'

def run_hourly():
    location = 'tauranga'
    version = 'v7'
    param_epoch = 37

    cfg = train_spg.get_config('base_hourly', version, location)
    data = data_utils.load_nc(cfg.input_file)

    stats = data_loader.open_stats(cfg.stats_path)
    param_path = get_params_path(cfg, param_epoch)

    preds = run_spg_mlp(data, param_path, stats=stats, cfg=cfg, freq='H')
    data_utils.save_nc_tprime(preds, cfg.ens_path / (location + '.nc'))

    run_pool(cfg=cfg, data=data, stats=stats, params_path=param_path, freq='H')

def run_daily():
    # TODO: Update with config file
    fname_obs = 'dunedin'
    version = 'v3'
    output_path = Path('/mnt/temp/projects/otago_uni_marsden/data_keep/spg/')
    save_obs = True
    feat_samples = 8

    output_path_obs = output_path / 'station_data' 
    output_path_ens = output_path / 'ensemble' / version
    output_path_ens.mkdir(exist_ok=True, parents=True)

    data = data_utils.load_data()#'/mnt/datasets/NationalClimateDatabase/NetCDFFilesByVariableAndSite/Hourly/Precipitation/1962.nc')
    param_path = f'params_daily_{version}.data'
    stats = data_loader.open_stats('./stats.json')
    
    run_kwargs = dict(data=data, stats=stats, params_path=param_path, feat_samples=feat_samples, freq='D')
    
    preds = run_spg_mlp(**run_kwargs)
    data_utils.save_nc_tprime(preds, output_path_ens / (fname_obs + '.nc'), units='mm/day')
    
    run_pool(output_folder=output_path_ens, file_pre=fname_obs, **run_kwargs)

if __name__ == '__main__':
    run_hourly()