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

N_CPU = 20

def fit_spg(data, use_tf=True, ar_depth=2, thresh=0.1):
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
              plot_folder=Path('./'), tprime=None):

    times = pd.date_range(start=start_date, end=end_date)
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

    if plot_folder:
        plot_qq(data, predictions, output_path=plot_folder / 'qq.png')

    return pd.Series(predictions, index=times)

def gen_save(arg, data, sp, df_magic, output_path):
    sce, ens_num, idx = arg
    sp.rnd_key = random.PRNGKey(seed=971*(int(idx)+42))

    predictions = gen_preds(sp, data, tprime=df_magic[sce])
    data_utils.make_nc(predictions, output_path / f'dunedin_{sce}_{str(ens_num).zfill(3)}.nc', tprime=df_magic[sce])

def run_non_stationary(output_path, data, scenario=['rcp26','rcp45','rcp60','rcp85','ssp119','ssp126',
                                                    'ssp245','ssp370','ssp434','ssp460','ssp585'], num_ens=10,):

    df_magic = data_utils.load_magic()
    t_prime = data_utils.get_tprime_for_times(data.index, df_magic['ssp245'])
    sp = fit_spg_ns(data, t_prime)

    ens = list(product(scenario, np.arange(num_ens) + 1))
    idxs = np.arange(len(ens)) 
    ens = np.concatenate([np.array(ens), idxs[:, None]], axis=1)

    print(f'Running {len(ens)} scenarios')

    with Pool(N_CPU) as p:
        p.map(partial(gen_save, data=data, sp=sp, df_magic=df_magic, output_path=output_path), ens)

def test_fit(data):
    sp = fit_spg(data)
    preds = gen_preds(sp, data, start_date='1950-1-1', end_date='1960-1-1')
    print(preds)

if __name__ == '__main__':
    fpath = "/mnt/temp/projects/emergence/data_keep/station_data/dunedin_btl_gardens_precip.tsv"
    fname_obs = 'dunedin.nc'
    fname_ens = fname_obs.replace('.nc', '_001.nc')

    output_path = Path('/mnt/temp/projects/otago_uni_marsden/data_keep/spg/')
    output_path_obs = output_path / 'station_data'
    output_path_ens = output_path / 'ensemble'

    data = data_utils.load_data(fpath)
    #run_non_stationary(output_path_ens, data)
    test_fit(data)
    #data_utils.make_nc(data, output_path_obs / fname_obs)
    #spg_tf = fit_spg(data, use_tf=True)
    #spg_ss = fit_spg(data, use_tf=False)
    #preds = gen_preds(spg_tf, data)

    #predictions = pd.Series(preds, index=data.index)
    #data_utils.make_nc(predictions, output_path_ens / fname_obs)
