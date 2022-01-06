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


def load_data(fpath="/mnt/temp/projects/emergence/data_keep/station_data/dunedin_btl_gardens_precip.tsv"):
    df = pd.read_csv(fpath, sep='\t', parse_dates=['Date(UTC)'], skiprows=8)

    df['date'] = pd.to_datetime(df['Date(UTC)'].values, format='%Y%m%d:%H%M')
    return pd.Series(df['Amount(mm)'].values, index=df['date'].values)


def fit_spg(data, use_tf=False, ar_depth=2, thresh=0.1):
    rd = distributions.RainDay(thresh=thresh, ar_depth=ar_depth)

    if use_tf:
        rain_dists = {0: distributions.TFWeibull(),
                      0.99: distributions.TFGeneralizedPareto()}
    else:
        rain_dists = {0: distributions.SSWeibull(),
                      0.99: distributions.SSGeneralizedPareto()}
    rng = random.PRNGKey(42)

    sp = SPG(rd, rain_dists, rng)
    sp.fit(data.values)
    sp.print_params()

    return sp

def plot_qq(target, predictions, output_path):
    plt.figure(figsize=(12, 8))
    qqplot(target, predictions, linewidth=0)
    plt.xlabel('Observations [mm/day]')
    plt.ylabel('Predictions [mm/day]')
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.savefig(output_path,  dpi=250)
    plt.close()


def gen_preds(sp: SPG, data: pd.Series, num_steps=None, plot_folder=Path('./')):
    if num_steps is None:
        num_steps = len(data)

    rd = sp.rainday
    cond = {'rain': None, 'rainday': jnp.array([[1]*rd.ar_depth])}
    #time = pd.date_range(start=data.index[0], end='2050-1-1')
    predictions = sp.generate(num_steps=num_steps, cond_init=cond)
    print(f'{(predictions >= rd.thresh).sum()/predictions.size} expected {(data >= rd.thresh).sum()/data.size}')

    if plot_folder:
        plot_qq(data, predictions, output_path =plot_folder / 'qq.png')

    return predictions


def make_nc(data, output_path):
    da = xr.DataArray.from_series(data)
    da.name = 'precipitation'
    da.attrs['units'] = 'mm/day'
    da = da.rename({'index': 'time'})

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

    spg_tf = fit_spg(data, use_tf=True)
    #spg_ss = fit_spg(data, use_tf=False)
    preds = gen_preds(spg_tf, data)

    predictions = pd.Series(preds, index=data.index)
    make_nc(predictions, output_path_ens / fname_obs)
