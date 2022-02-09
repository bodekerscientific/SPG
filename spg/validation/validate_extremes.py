#%%
from bslibs.regression import gev
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import statsmodels.api as sm
import pandas as pd

from spg.data_utils import load_magic, load_wh, get_tprime_for_years
from spg.validation.data_sources import load_all_models
from multiprocessing import Pool

base_path = Path('/mnt/temp/projects/otago_uni_marsden/data_keep/')
output_path = base_path / 'plots' / 'reg_plots'
output_path.mkdir(parents=True, exist_ok=True)

loc_name = 'dunedin'
location_lat = -44.526
location_lon = 169.889

obs_path = ens_path = base_path / f'spg/station_data/{loc_name}.nc'
wh_path = base_path / 'weather_at_home' / loc_name


def fit_and_plot(x, y, title, **kwargs):
    idx   = np.argsort(x)
    x = x[idx]
    y = y[idx]

    X = sm.add_constant(x)

    mod = sm.OLS(y, X)
    res = mod.fit()

    title += f' {(res.params[1]/y.mean())*100:.3f}% increase per degree'

    predictions = res.get_prediction()
    sum_df = predictions.summary_frame(alpha=0.05) 

    plot_reg_fit(x, y, sum_df['mean'], params=res.params, title=title,
                ci_upper=sum_df['mean_ci_upper'], ci_lower=sum_df['mean_ci_lower'], **kwargs)

def plot_reg_fit(x, y, fit, params=None, title=None, ci_upper=None, ci_lower=None, ax=None, figsize=(12, 8), label_dict={}):
    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    if title is not None:
        ax.set_title(title)

    fit_label = 'Fit'
    if params is not None:
        fit_label += f" {params[1]:.3f}*x + {params[0]:.3f}"

    ax.plot(x, y, "o", label="Obs")
    ax.plot(x, fit, "b-", label=fit_label)
    if ci_upper is not None:
        ax.plot(x, ci_upper, "r--", label='Fit 95% CI')
        ax.plot(x, ci_lower, "r--")

    ax.legend(loc="best")
    #ax.set_xlim(x.min(), x.max())

    return ax


# wh_batches = ['batch_870_ant', 'batch_871_ant', 'batch_872_ant']


#%%
rcp_mappings = {
    'rcp85' : 'RCP8.5',
    'rcp60' : 'RCP6.0',
    'rcp45' : 'RCP4.5',
    'rcp26' : 'RCP2.6'
}

df_magic = load_magic()
#%%
rcm_output = defaultdict(list)
for rcp in df_magic.columns:
    if rcp in rcp_mappings:
        print(f'Processing rcp {rcp}')
        rcm_data = load_all_models(location_lat, location_lon, rcp_mappings[rcp])
        for k,v in rcm_data.items():
            tp = get_tprime_for_years(v['year'].values, df_magic[rcp])
            rcm_output['tprime'].extend(tp)
            rcm_output['precipitation'].extend(v['rain'].values)
            rcm_output['model'].extend([k]*len(tp))
            rcm_output['year'].extend(v['year'].values)
            rcm_output['rcp'].extend([rcp]*len(tp))

rcm_output = pd.DataFrame(rcm_output)

#%%
def load_bmax(path):
    with xr.open_dataset(path) as ds:
        tp = ds['tprime'].resample(time='1D').mean().values.reshape(-1)
        pr = ds['precipitation'].resample(time='1D').sum()
        ds = pd.DataFrame({'tprime' : tp, 'precipitation' : pr.values}, 
                            index=pr.time.values)
        
        ds_grp = ds.groupby(ds.index.year)
        mask = ds_grp.count()['precipitation'].values > 360
        
        return ds_grp.max()[mask]

def load_ds_bmax_mf(paths):
    with Pool(48) as p:
        output = p.map(load_bmax, paths, chunksize=1)

    return pd.concat(output)

def show_and_save(path=None):
    plt.ylabel('Precipitation [mm/day]')
    plt.xlabel('SH Land Temperature Anomaly [K]')
    if path is not None:
        plt.savefig(path, dpi=250, transparent=False)
    plt.show()

#%%
for version in ['v6']:
    ens_path = base_path / f'spg/ensemble_hourly/{version}/'
    ens_all = list(ens_path.glob('dunedin_*.nc'))
    ds_ens = load_ds_bmax_mf(ens_all)
    
    fit_and_plot(ds_ens['tprime'].values.reshape(-1),
                ds_ens['precipitation'].values.reshape(-1), title=f'{version}_hourly Annual Daily Maxima')
    show_and_save(output_path / f'spg_hourly_{version}.png')

#%%
for model in rcm_output['model'].unique():
    subset = rcm_output[rcm_output['model'] == model]
    fit_and_plot(subset['tprime'].values, subset['precipitation'].values, f'RCM {model}')
    show_and_save(output_path / f'rcm_{model}.png')

for version in ['v0.2', 'v3', 'v4', 'v5_wh']:
    ens_path = base_path / f'spg/ensemble/{version}/'
    ens_all = list(ens_path.glob('dunedin_*.nc'))
    # print(f'Loading {ens_all}')
    ds = load_ds_bmax_mf(ens_all)
    fit_and_plot(y=ds['precipitation'].values, x=ds['tprime'].values, title=f'{version} Annual Daily Maxima')
    show_and_save(output_path / f'spg_daily_{version}.png')

#%%
wh_all = load_wh(num_ens=500)
wh_bmax = pd.DataFrame([wh.max() for wh in wh_all])

fit_and_plot(wh_bmax['tp'].values, wh_bmax['pr'].values, f'wether@home Annual Daily Maxima')
show_and_save(output_path / 'wh.png')

#%%
obs_ds = xr.open_dataset(obs_path)
obs_ds = obs_ds.groupby(obs_ds.time.dt.year)
obs_ds = obs_ds.max().isel(year=obs_ds.count()['precipitation'] > 350)
fit_and_plot(obs_ds['tprime'].values, obs_ds['precipitation'].values, f'Observations Annual Daily Maxima')
show_and_save(output_path / 'obs.png')
# %%
