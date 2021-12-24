from spg.generator import SPG
from spg import distributions 
from bslibs.plot.qqplot import qqplot

import pandas as pd
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import xarray as xr
from bslibs.ncutils import get_attributes
from pathlib import Path

def load_data(fpath):
    df = pd.read_csv(fpath, sep='\t', parse_dates=['Date(UTC)'], skiprows=8)

    df['date'] = pd.to_datetime(df['Date(UTC)'].values, format='%Y%m%d:%H%M')
    return pd.Series(df['Amount(mm)'].values, index=df['date'].values)


def fit_and_generate(data, num_steps=None, plot_folder = Path('./')):
    rd = distributions.RainDay(thresh=0.1, ar_depth=2)
    rain_dists = {0 : distributions.SSWeibull(),
                  0.994 : distributions.SSGPD()}
    rng = random.PRNGKey(42)

    sp = SPG(rd, rain_dists, rng)
    sp.fit(data.values)

    if num_steps is None:
        num_steps = len(data)

    cond = {'rain' : None, 'rainday' : jnp.array([[1, 1]])}
    #time = pd.date_range(start=data.index[0], end='2050-1-1')
    predictions = sp.generate(num_steps=num_steps, cond_init=cond)
    print(f'{(predictions >= rd.thresh).sum()/predictions.size} expected {(data >= rd.thresh).sum()/data.size}')

    plt.figure(figsize=(12, 8))
    qqplot(data, predictions, linewidth=0)
    plt.xlabel('Observations [mm/day]')
    plt.ylabel('Predictions [mm/day]')
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.savefig(plot_folder / 'qq.png',  dpi=250)
    
    return predictions

def make_nc(data, output_path):
    da = xr.DataArray.from_series(data)
    da.name = 'precipitation'
    da.attrs['units'] = 'mm/day'
    da = da.rename({'index' : 'time'})

    ds = xr.Dataset()
    ds['precipitation'] = da
    ds.attrs = get_attributes()
    
    print(f'Saving to {output_path}')
    ds.to_netcdf(output_path)


if __name__ == '__main__':
    fpath = "/mnt/temp/projects/emergence/data_keep/station_data/dunedin_btl_gardens_precip.tsv"
    fname_obs = 'dunedin.nc'
    fname_ens = fname_obs.replace('.nc', '_001.nc')
    
    output_path = Path('/mnt/temp/projects/otago_uni_marsden/data_keep/spg/')
    output_path_obs = output_path / 'station_data'
    output_path_ens = output_path / 'ensemble'

    data = load_data(fpath)
    make_nc(data, output_path_obs / fname_obs)

    predictions = fit_and_generate(data)
    predictions = pd.Series(predictions, index=data.index)
    make_nc(predictions, output_path_ens / fname_obs)
