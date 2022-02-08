from collections import defaultdict
from functools import lru_cache
import pandas as pd
import xarray as xr
from bslibs.ncutils import get_attributes
from pathlib import Path
import random

@lru_cache
def load_magic(target_path='data/magic_tprime_sh_land.csv'):
    return pd.read_csv(target_path, parse_dates=['date'], index_col='date')

def get_tprime_for_times(dates, tprime_df):
    dates = pd.DatetimeIndex(dates)
    years = dates.year
    temp = pd.Series(tprime_df.values, index=tprime_df.index.year)
    return temp.loc[years].values

def make_nc(data, output_path, tprime=None, units='mm/day'):
    da = xr.DataArray.from_series(data)
    da.name = 'precipitation'
    da.attrs['units'] = units
    if 'index' in da.dims:
        da = da.rename({'index': 'time'})

    ds = xr.Dataset()
    ds['precipitation'] = da
    if tprime is not None:
        ds['tprime'] = xr.DataArray(tprime, dims=('time',))
        ds['tprime'].attrs = {
            'description' : 'Global southern hemisphere land temperature anomaly',
            'source' : '/mnt/temp/projects/otago_uni_marsden/data_keep/spg/MAGICC/magicc_runs_20210630.csv',
            'reference_year' : 1765,
            'units' : 'K'
        }
    ds.attrs = get_attributes()

    print(f'Saving to {output_path}')
    ds.to_netcdf(output_path)

def load_data(fpath="/mnt/temp/projects/emergence/data_keep/station_data/dunedin_btl_gardens_precip.tsv"):
    df = pd.read_csv(fpath, sep='\t', parse_dates=['Date(UTC)'], skiprows=8)

    df['date'] = pd.to_datetime(df['Date(UTC)'].values, format='%Y%m%d:%H%M')
    out = pd.Series(df['Amount(mm)'].values, index=df['date'].values)
    out.index = out.index.normalize()
    return out

def load_data_hourly(fpath="/mnt/datasets/NationalClimateDatabase/NetCDFFilesByVariableAndSite/Hourly/Precipitation/5212.nc"):
    ds = xr.open_dataset(fpath)
    # Select only the hourly values
    ds = ds.sel(time=ds.period.values == 1)
    return ds['precipitation'].to_series()

def load_wh(base_path='/mnt/temp/projects/otago_uni_marsden/data_keep/weather_at_home/dunedin/', 
            batches= ['batch_870_ant', 'batch_871_ant', 'batch_872_ant'], num_ens=400, spin_up_days=8,
            mult_factor=1.75):

    batches_tprime = {'batch_870_ant' : 1.5, 'batch_871_ant' : 2.0, 'batch_872_ant' : 3.0}

    random.seed(42)

    out = []
    for batch in batches:
        files = list((Path(base_path) / batch).glob('*.nc'))
        random.shuffle(files)
        count = 0
        for f in files:
            
            with xr.open_dataset(f) as ds:
                if len(ds['time1']) == 600:
                    ds = ds.isel(time1=slice(600-360-spin_up_days, 600), z0=0)
                    df = pd.DataFrame({'pr' : ds['precipitation'].values*24*60*60*mult_factor, 'dts' : ds['time1'].values,
                                       'tp'  :  batches_tprime[batch]})
                    out.append(df)
                    count += 1
           
            if count == num_ens:
                break

    return out

if __name__ == '__main__':
    load_data_hourly()
    #print(t_prime)