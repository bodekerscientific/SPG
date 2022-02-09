import xarray as xr
import pandas as pd
from pathlib import Path

rcm_dir = Path("/mnt/datasets/RCMData/Version6/")


def load_and_sel(files, lat, lon):
    unmerged_datasets = []
    for file in files:
        dataset = xr.open_dataset(file)
        subset = dataset.sel(latitude=lat, longitude=lon, method="nearest")
        unmerged_datasets.append(subset)
    merged_datasets = xr.merge(unmerged_datasets)
    return merged_datasets


def load_all_models(location_lat, location_lon, pathway="RCP8.5", block_maximia=True):
    models = [d.parts[-1]
              for d in (rcm_dir / pathway).glob("*/") if d.is_dir()]

    rcm_files = {}
    for model in models:
        rcm_files[model] = sorted(
            (rcm_dir / pathway / model).glob("TotalPrecipCorr*_permuted.nc")
        )
        # Some RCMs have no permuted files.
        if len(rcm_files[model]) == 0:
            rcm_files[model] = sorted(
                (rcm_dir / pathway / model).glob("TotalPrecipCorr*.nc")
            )

    datasets = {}
    for model in rcm_files.keys():
        print(f"Loading model {model}")
        ds = load_and_sel(
            rcm_files[model],
            lat=location_lat,
            lon=location_lon
        )
        if block_maximia:
            grp = ds.groupby(ds.time.dt.year)
            mask = grp.count()['rain'].values > 360
            ds = grp.max().isel(year=mask)
        
        datasets[model] = ds

    return datasets
