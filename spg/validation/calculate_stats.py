#%%
"""
    Calculate statistics for the non-stationary spg.
"""
from collections import defaultdict
from functools import partial
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from multiprocessing.pool import ThreadPool as Pool
from pathlib import Path
from spg.data_utils import locations, load_nc
import scipy.stats as ss

def group_max(df_grp, ds):
    return df_grp.max()['precipitation']

def group_mean(df_grp, ds, mult=365):
    # Use mean as this will be more accurate with missing data
    return df_grp.mean()['precipitation']*mult

def group_year(df_grp, ds):
    return df_grp.mean().index

def rain_days(df_grp, ds : pd.DataFrame, thresh=1.0):
    ds = ds >= thresh
    ds_grp = ds.groupby(ds.index.year)
    return ds_grp.mean()['precipitation']

def calculate_annual_metrics(ds, func, min_count):
    ds_grp = ds.groupby(ds.index.year)
    mask = ds_grp.count()['precipitation'].values > min_count
    return func(ds_grp, ds)[mask]


def load_ds(path, ann_funcs : dict, min_count):
    with xr.open_dataset(path) as ds:
        pr = ds['precipitation']#.resample(time='1D').sum()
        ds = pd.DataFrame({'precipitation' : pr.values},  index=pr.time.values)
    
        ds_ann = {}
        for name, func in ann_funcs.items():
            ds_ann[name] = calculate_annual_metrics(ds, func, min_count)
        year = ds_ann.pop('year')
        
    return ds, pd.DataFrame(ds_ann, index=year)

def split_combines_dfs(results):
    df_all, df_ann_all = [], []
    for df, df_ann in results:
        df_all.append(df)
        df_ann_all.append(df_ann)
    return pd.concat(df_all).reset_index(),  pd.concat(df_ann_all).reset_index()

def load_ds_mf(paths, ann_funcs : dict, min_count):
    with Pool(24) as p:
        results = p.map(partial(load_ds, ann_funcs=ann_funcs, min_count=min_count), paths, chunksize=1)

    return split_combines_dfs(results)

#%%
epoch_mapping = {
    'daily' : {
        'auckland' : '020',
        'christchurch' : '005',
        'dunedin' : '016',
        'tauranga' : '036'
    },
    'hourly' : {
        'auckland' : '010',
        'christchurch' : '008',
        'dunedin' : '009',
        'tauranga' : '003'
    }
}

np.random.seed(42)

#%%
freq = 'hourly'
is_station = False

min_count = 360 if freq == 'daily' else 360*24
load_station = False

base_path = Path(f'/mnt/temp/projects/otago_uni_marsden/data_keep/spg')
ens_path = base_path / f'ensemble_{freq}_paper/v10/'
station_path = base_path / f'station_data{"_hourly" if freq == "hourly" else ""}'

output_path = base_path / 'stats'
output_path.mkdir(parents=False, exist_ok=True)

#%%
ann_funcs = {
    'total' : partial(group_mean, mult=24*365 if freq == 'hourly' else 365),
    'max' : group_max,
    'wet_days' : partial(rain_days, thresh=0.1 if freq == 'hourly' else 1.0),
    'year' : group_year,
}
#%%
dfs = {}
dfs_annual = {}

for loc in locations:
    if is_station:
        df, df_annual = load_ds(station_path / f'{loc}.nc', ann_funcs, min_count)
    else:
        ensembles = list((ens_path / f'{loc}_epoch_{epoch_mapping[freq][loc]}').glob(f'{loc}*.nc'))
        df, df_annual = load_ds_mf(ensembles, ann_funcs, min_count)
    
    dfs[loc] = df 
    dfs_annual[loc] = df_annual
#%%

def get_gev_bootstrap(data, num_fits, probs):
    params_original = ss.genextreme.fit(data)
    results = []
    for i in range(num_fits):
        # Generate a new sample from the GEV fit of the original sample
        block_maxima_generated = ss.genextreme.rvs(*params_original, size=len(data))
        # Create a new GEV fit for the new sample
        params = ss.genextreme.fit(block_maxima_generated)
        results.append(ss.genextreme.ppf(probs, *params))
    
    return np.array(results)

def bmax_quantile(data, probs):
    output = []
    for p in probs:
        output.extend([np.nan, np.quantile(data, p), np.nan])
    return np.array(output)

#%%
gev_columns = ['5%', 'median', '95%']
def add_year(columns, n):
    return [f'{c} {n}yr' for c in columns]
gev_columns = add_year(gev_columns, 10) + add_year(gev_columns, 100)

#%%
# Calculate 
# Calculate, Annual Mean, Annual variability,  10 year maximum, 100 year maximum, Skewness, Annual Rain days, 
#%%
output_df = defaultdict(list)
for loc in locations:
    print(f'processing {loc}')
    probs = [1.0-1/10, 1-1/100]
    data = dfs_annual[loc]['max']
    
    if is_station:
        results = get_gev_bootstrap(data, 500, probs)
        results = np.quantile(results, [0.05, 0.5, 0.95], axis=0).T.reshape(-1)
    else:
        results = bmax_quantile(data, probs)
    
    for n, v in zip(gev_columns, results):
        output_df[n].append(v)

# %%
len(data)
#%%
output_df = pd.DataFrame(output_df, index=locations) 
output_df.to_csv(output_path / f'extremes_{freq}_{"station" if is_station else "spg"}.csv', float_format='%.2f')
with open(output_path / f'extremes_latex_{freq}_{"station" if is_station else "spg"}.txt', 'w') as f:
    output_df.to_latex(f, float_format="%.2f", na_rep='-')
#%%
output_df
#%%
z_val = 1.645 # 90% CI

output_metrics = defaultdict(list)

for loc in locations:
    print(f'processing {loc}')

    df_ann = dfs_annual[loc]
    df = dfs[loc]
    
    output_metrics['Annual precipitation'].append(df_ann['total'].mean())
    output_metrics['Annual precipitation error'].append(ss.sem(df_ann['total'])*z_val)
    output_metrics['Annual precipitation std'].append(df_ann['total'].std())
    
    output_metrics['Wet proportion'].append(df_ann['wet_days'].mean()*100)
    output_metrics['Wet proportion error'].append(ss.sem(df_ann['wet_days']*100)*z_val)
    
    output_metrics['Skewness'].append(ss.skew(df['precipitation'].values))
    output_metrics['Kurtosis'].append(ss.kurtosis(df['precipitation'].values))
    
    
#%%    
output_metrics = pd.DataFrame(output_metrics, index=locations) 
output_metrics.to_csv(output_path / f'metrics_{freq}_{"station" if is_station else "spg"}.csv', float_format='%.2f')# %%
with open( output_path / f'metrics_latex_{freq}_{"station" if is_station else "spg"}.txt', 'w') as f:
    output_metrics.to_latex(f, float_format="%.2f")

# %%
output_metrics
# %%
