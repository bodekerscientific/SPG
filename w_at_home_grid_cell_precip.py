import numpy as np
import xarray as xr
from glob import glob
import sys
from bslibs import env
import os

""" Script extracts rainfall values for given locations from weather at home ensembles """

def euclidian_distance(lat1, lat2, lon1, lon2):
    '''
    Calculates euclidian distance between twp pairs of co-ordinates
    '''
    return (abs(lat1 - lat2) ** 2 + abs(lon1 - lon2) ** 2) ** 0.5

def get_closest(lat, lng, file_path_list):
    with xr.open_dataset(file_path_list[np.random.randint(0,len(file_path_list))]) as ds:
        #land_sea_mask = np.isnan(ds.tmax.values[0,0,:,:])
        lat_grid = ds.variables['global_latitude0'].values
        lng_grid = ds.variables['global_longitude0'].values
        distances = euclidian_distance(lat_grid, lat, lng_grid, lng)
        #distances[land_sea_mask] = np.nan
        inds = np.unravel_index(np.nanargmin(distances),distances.shape) 
    return inds

if __name__=="__main__":

    ensembles = ["batch_870_ant","batch_871_ant","batch_872_ant"] #Add to list 
    latitude = -45.8741
    longitude = 170.5035
    out_path = "/mnt/temp/projects/otago_uni_marsden/data_keep/weather_at_home/dunedin"

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for ens in ensembles: 
        file_path_list = glob(os.path.join(env.datasets(''),'weatherathome','processed',ens,"*.nc"))
        inds = get_closest(latitude,longitude,file_path_list) 

        if not os.path.exists(f"{out_path}/{ens}"):
            os.mkdir(f"{out_path}/{ens}")

        for fl in file_path_list: 
            with xr.open_dataset(fl) as ds: 
                ds = ds.precipitation.isel(latitude0=inds[0],longitude0=inds[1])
            ds.to_netcdf(f"{out_path}/{ens}/{fl.split('/')[-1]}")
