#%%
import xarray as xr
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from spg.data_utils import save_nc_tprime

input_path = Path('/mnt/datasets/NationalClimateDatabase/NetCDFFilesByVariableAndSite/Hourly/Precipitation/')

locations = {
    'dunedin' : (-45.89, 170.501),
    'christchurch' : (-43.530150, 172.635164),
    'tauranga' : (-37.687027, 176.165421)
}

def load_close_stations(files, target_lat, target_lon, max_dist=40, max_height=70):
    out = {}
    for fname in files:
        with xr.open_dataset(fname) as ds:
            dist = (ds.longitude.values - target_lon)**2 + (ds.latitude.values - target_lat)**2
            dist = 111*dist**0.5
            if ds.station_height.values.size == 1:
                if dist < max_dist and ds.station_height.values < max_height:
                    out[ds.attrs["site name"]] = ds.load()
                    print(f'Found station {ds.attrs["site name"]} with {len(ds.precipitation)} values, from {ds.time.min().dt.date.values} to {ds.time.max().dt.date.values} and distance {dist:.1f}km') 
    return out
    # # Select only the hourly values
    # ds = ds.sel(time=ds.period.values == 1)
    # return ds['precipitation'].to_series()


def to_series(ds):
    # Important, remove non-hourly values
    ds = ds.sel(time=ds.period.values == 1)
    return ds['precipitation'].to_series()

if __name__ == '__main__':
    files = list(input_path.glob('*.nc'))

    # Load the close stations
    all_sites = {}
    for site_name, (t_lat, t_lon) in locations.items():
        print(site_name)
        all_sites[site_name] = load_close_stations(files, t_lat, t_lon)

    # Join the two dunedin airport AWS
    a = to_series(all_sites['dunedin']['DUNEDIN AERO AWS'])
    b = to_series(all_sites['dunedin']['DUNEDIN AERO'])

    # Manually select the most suitable staions
    dunedin = pd.concat([b, a])
    chch = to_series(all_sites['christchurch']['CHRISTCHURCH AERO'])
    tauranga = to_series(all_sites['tauranga']['TAURANGA AERO AWS'])

    #%%
    plt.figure(figsize=(20, 12))
    plt.hist((dunedin, chch, tauranga), 15,
            label=('dun', 'chch', 'tauranga'), density=True)
    plt.yscale('log')
    plt.legend()
    plt.savefig('hist.png')
    plt.close()

    # Save the stations
    output_path = Path('/mnt/temp/projects/otago_uni_marsden/data_keep/spg/')    
    output_path_obs = output_path / 'station_data_hourly' 

    save_nc_tprime(dunedin, output_path_obs / 'dunedin.nc')
    save_nc_tprime(chch, output_path_obs / 'christchurch.nc')
    save_nc_tprime(tauranga, output_path_obs / 'tauranga.nc')

