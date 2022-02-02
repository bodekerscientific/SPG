from bslibs.plot.qqplot import qqplot

import pandas as pd
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool

from spg.generator import SPG
from spg import distributions, jax_utils, data_utils, generator
from pathlib import Path
from itertools import product
import numpy as np

import data_loader
import spg_dist
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
                                                    'ssp245','ssp370','ssp434','ssp460','ssp585'], num_ens=1,
                                                    **kwargs):
    
    df_magic = data_utils.load_magic()
    t_prime = data_utils.get_tprime_for_times(data.index, df_magic['ssp245'])
    sp = fit_spg_ns(data, t_prime)

    ens = list(product(scenario, np.arange(num_ens) + 1))
    idxs = np.arange(len(ens)) 
    ens = np.concatenate([np.array(ens), idxs[:, None]], axis=1)

    print(f'Running {len(ens)} scenarios')

    with Pool(N_CPU) as p:
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

def run_spg_mlp(data : pd.Series, params_path : Path, feat_samples = 6*24, spin_up_hours=1000):
    # Number of samples required for feature calculation
    
    rng = random.PRNGKey(np.random.randint(1e10))
    
    model = spg_dist.get_model()
    params = load_params(model, params_path)
    
    pr_re = data.resample('H').asfreq()
    dts = pd.date_range(start=pr_re.index.min() - pd.Timedelta(hours=spin_up_hours), end=pr_re.index.max(), freq='H')
    
    # Find the first index where we get non nan values
    idx_valid = np.where(~np.isnan(pr_re.rolling(feat_samples+1).mean().values))[0][0]
    output = pd.Series(np.zeros(len(dts), dtype=np.float32), index=dts)

    # Copy of the initial values for
    output.iloc[0:feat_samples] = pr_re[idx_valid-feat_samples+1:idx_valid+1].values 

    @jax.jit
    def sample_func(x, rng):
        return model.apply(params, x, rng, method=model.sample)

    idx = 0
    for n in tqdm(range(feat_samples, len(output))):
        subset_cond = output.iloc[n - feat_samples:n+1]
        x, y = data_loader.generate_features(subset_cond)

        assert len(x) == 1
        assert y == 0
        
        x = x[0]

        rng, sample_rng = random.split(rng)
        x_next = sample_func(x, sample_rng)
        output.iloc[n] = x_next

    # Remove the real data from the output
    output = output.iloc[spin_up_hours:]

    return output
    
def save_nc_tprime(data, output_path):
    df_magic = data_utils.load_magic()
    t_prime = data_utils.get_tprime_for_times(data.index, df_magic['ssp245'])
    data_utils.make_nc(data, output_path, tprime=t_prime)

if __name__ == '__main__':
    
    fname_obs = 'dunedin.nc'
    version = 'v3'
    output_path = Path('/mnt/temp/projects/otago_uni_marsden/data_keep/spg/')
    save_obs = True
    
    output_path_obs = output_path / 'station_data_hourly' 
    output_path_ens = output_path / 'ensemble_hourly' / version
    output_path_ens.mkdir(exist_ok=True, parents=True)

    data = data_utils.load_data_hourly()
    if save_obs:
        save_nc_tprime(data, output_path_obs / fname_obs)

    scale = data_loader.get_scale(data)
    
    preds = run_spg_mlp(data/scale, 'params_069.data')*scale
    save_nc_tprime(data, output_path_ens / fname_obs)
