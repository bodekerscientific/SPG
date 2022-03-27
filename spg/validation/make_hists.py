#%%
"""
    
    Makes historgrams / qqplots of the SPG vs historical obs

"""

from pathlib import Path
import matplotlib.pyplot as plt
from bslibs.plot.qqplot import qqplot
import xarray as xr
import pandas as pd

base_path = Path('/mnt/temp/projects/otago_uni_marsden/data_keep/')


# %%
def plot(obs, target, title, log=True, density=True, xlim=(0, 30)):
    obs = obs[obs > 0.1]
    target = target[target > 0.1]
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    qqplot(obs, target, ax=ax2, square=False)
    
    ax1.hist((obs, target), 15, label=('Obs', 'SPG'), density=density)
    
    if log:
        ax1.set_yscale('log')

    ax1.set_ylabel('Density' if density else 'Count')
    
    ax1.set_xlabel('Precipitation [mm]')
    ax1.set_title(title + ' Hist')

    ax2.set_xlabel('Obs Precipitation [mm]')
    ax2.set_ylabel('SPG Precipitation [mm]')
    ax2.set_title(title + ' QQ plot')
    ax2.set_xlim(0)
    ax2.set_ylim(0)
    
    ax1.set_xlim(0)
    ax1.set_ylim(0)
    # for ax in [ax1, ax2]:
    #     ax.set_xlim(*xlim)
   
    ax1.legend()

def calc_bmax(ds):
    ds_grp = ds.groupby(ds.time.dt.year)
    count = ds_grp.count()['precipitation'].values
    max_count =float(max(count))
    # Need at least 90% of the data

    mask = ds_grp.count()['precipitation'].values/max_count > 0.9
        
    return ds_grp.max().isel(year=mask)

# %%
def resample(ds):
    ds = ds.reindex({'time' : pd.date_range(ds.time.values.min(), 
                                  ds.time.values.max(), freq='H')})

    ds = ds.resample({'time' : '1D'}).sum(skipna=True, min_count=24)
    return ds.dropna('time', subset=['precipitation'])
    

#%%
def make_plots(ds_obs, ds_spg, output_path, kind='hourly'):
    plot(ds_obs['precipitation'].values, ds_spg['precipitation'].values, 
        f'{kind} Precipitation'.title())
    plt.tight_layout()
    plt.savefig(output_path / f'hist_{kind}.png',dpi=200)
    plt.show()

    plot(calc_bmax(ds_obs)['precipitation'].values, calc_bmax(ds_spg)['precipitation'].values, 
        title=f'{kind} Annual Block Maximia Precipitation'.title(), log=False, density=False)
    plt.tight_layout()
    plt.savefig(output_path / f'hist_{kind}_bmax.png', dpi=200)
    plt.show()

if __name__ == '__main__':
    version = 'v9_split'

    for loc_dir in (base_path / f'spg/ensemble_split/{version}/').iterdir():
        print(loc_dir)
        loc_full = loc_dir.name
        loc = loc_full.split('_epoch')[0]
        print(loc)

        output_path = base_path / 'plots' / 'hist_plots' / version / loc_full
        output_path.mkdir(parents=True, exist_ok=True)
        
        obs_path = ens_path = base_path / f'spg/station_data_hourly/{loc}.nc'
        spg_path = loc_dir / f'{loc}.nc'

        if obs_path.exists() and spg_path.exists():
            ds_obs = xr.open_dataset(obs_path).load()
            ds_spg = xr.open_dataset(spg_path).load()

            make_plots(ds_obs, ds_spg, output_path)

            make_plots(resample(ds_obs), resample(ds_spg), output_path, kind='daily')
        else:
            print(f'Skipping {loc_full} as cant find files')
# %%

# %%
