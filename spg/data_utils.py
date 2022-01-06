import pandas as pd
import xarray as xr
from bslibs.ncutils import get_attributes
from pathlib import Path

def load_magic(target_path='data/magic_tprime_sh_land.csv'):
    return pd.read_csv(target_path, parse_dates=['date'], index_col='date')

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

def load_data(fpath="/mnt/temp/projects/emergence/data_keep/station_data/dunedin_btl_gardens_precip.tsv"):
    df = pd.read_csv(fpath, sep='\t', parse_dates=['Date(UTC)'], skiprows=8)

    df['date'] = pd.to_datetime(df['Date(UTC)'].values, format='%Y%m%d:%H%M')
    return pd.Series(df['Amount(mm)'].values, index=df['date'].values)
