"""
    Evaluates the SPG and generates ensambles

    @Author Leroy Bird
    
    You might need to set, as TF and JAX will both try grab the full gpu memory.
        export XLA_PYTHON_CLIENT_PREALLOCATE=false

"""
from bslibs.plot.qqplot import qqplot

import pandas as pd
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import get_context

from spg.generator import SPG
from spg import distributions, jax_utils, data_utils, generator, data_loader, spg_dist
from pathlib import Path
from itertools import product
import numpy as np

import flax
from tqdm import tqdm
import jax

N_CPU = 20


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

def load_params(model, path, num_feat=17): 
    x = jnp.zeros((num_feat,), dtype=jnp.float32)
    params = model.init(random.PRNGKey(0), x, random.PRNGKey(42))

    with open(path, 'rb') as f:
        params = flax.serialization.from_bytes(params, f.read())

    return params
 
def setup_output(data, feat_samples, start_date, end_date, spin_up_hours=1000, freq='H'):
    pr_re = data.resample(freq).asfreq()
    dts = pd.date_range(start=start_date - pd.Timedelta(hours=spin_up_hours), end=end_date, freq=freq)
    
    # Find the first index where we get non nan values
    idx_valid = np.where(~np.isnan(pr_re.rolling(feat_samples+1).mean().values))[0][0]
    output = pd.Series(np.zeros(len(dts), dtype=np.float32), index=dts)
    output.index.name = 'time'

    # Copy of the initial values from the obs
    output.iloc[0:feat_samples] = pr_re[idx_valid-feat_samples+1:idx_valid+1].values 
    return output

def run_spg_mlp(data : pd.Series, params_path : Path, feat_samples = 6*24, spin_up_hours=1000, 
                start_date=None, end_date=None, rng=None, sce='ssp245', use_tqdm=True):
    if rng is None:
        rng = random.PRNGKey(np.random.randint(1e10))
    
    model = spg_dist.get_model()
    params = load_params(model, params_path)

    if start_date is None:
        start_date = data.index.min()
    if end_date is None:
        end_date = data.index.max()
    
    output = setup_output(data, feat_samples, start_date, end_date, spin_up_hours=spin_up_hours)

    @jax.jit
    def sample_func(x, rng):
        return model.apply(params, x, rng, method=model.sample)

    n_range = range(feat_samples, len(output))
    if use_tqdm:
        n_range = tqdm(n_range)

    for n in n_range:
        subset_cond = output.iloc[n - feat_samples:n+1]
        x, y = data_loader.generate_features(subset_cond, sce=sce)
        
        # Sometimes (very rarely) we get an np.nan
        # So just repeat until we get a valid output
        out = jnp.nan
        while(not jnp.isfinite(out)):
            rng, sample_rng = random.split(rng)
            out = sample_func(x[0], sample_rng)
        output.iloc[n] = out

    # Remove the real data and spin up period from the output
    return output.iloc[spin_up_hours:]
    
def save_nc_tprime(data, output_path,  units='mm/hr', sce='ssp245'):
    df_magic = data_utils.load_magic()
    t_prime = data_utils.get_tprime_for_times(data.index, df_magic[sce])
    print(f'Saving file {output_path}')
    data_utils.make_nc(data, output_path, tprime=t_prime, units=units)

def run_save_proj(ens_args, output_folder, file_pre, data : pd.Series, scale, params_path : Path, 
                  start_date=pd.Timestamp(1950, 1, 1), end_date=pd.Timestamp(2100, 1, 1), **kwargs):

    sce, ens_num, idx = ens_args
    rng = random.PRNGKey(seed=971*(int(idx)+42))
    preds = run_spg_mlp(data/scale, params_path, start_date=start_date, end_date=end_date, rng=rng, **kwargs)*scale

    output_path = output_folder / f'{file_pre}_{sce}_{str(ens_num).zfill(3)}.nc'    
    print(f'Saving to {output_path}')
    save_nc_tprime(preds, output_path, sce=sce)

def run_pool(scenarios=['rcp26','rcp45','rcp60','rcp85','ssp119','ssp126',
                        'ssp245','ssp370','ssp434','ssp460','ssp585'], num_ens=5, **kwargs):

    ens = list(product(scenarios, np.arange(num_ens) + 1))
    idxs = np.arange(len(ens)) 
    ens = np.concatenate([np.array(ens), idxs[:, None]], axis=1)

    print(f'Running {len(ens)} scenarios')

    with get_context('spawn').Pool(N_CPU) as p:
        p.map(partial(run_save_proj, use_tqdm=False, **kwargs), ens)

if __name__ == '__main__':
    fname_obs = 'dunedin'
    version = 'v3'
    output_path = Path('/mnt/temp/projects/otago_uni_marsden/data_keep/spg/')
    save_obs = True
    
    output_path_obs = output_path / 'station_data_hourly' 
    output_path_ens = output_path / 'ensemble_hourly' / version
    output_path_ens.mkdir(exist_ok=True, parents=True)

    data = data_utils.load_data_hourly()#'/mnt/datasets/NationalClimateDatabase/NetCDFFilesByVariableAndSite/Hourly/Precipitation/1962.nc')
    if save_obs:
        save_nc_tprime(data, output_path_obs / (fname_obs + '.nc'))

    scale = data_loader.get_scale(data)
    run_pool(output_folder=output_path_ens, file_pre=fname_obs, data=data, scale=scale, params_path='params_069.data')

    # preds = run_spg_mlp(data/scale, 'params_069.data')*scale
    # save_nc_tprime(preds, output_path_ens / (fname_obs + '.nc'))
