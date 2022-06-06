import pytest
import numpy as np
import xarray as xr
from spg.stationary_converter import StationaryConverter

def test_init():
    sc = StationaryConverter()
    
def create_dataset():
    # Create hourly dataset where all values are the same
    start_date = np.datetime64("2000-01-01")
    ds = xr.Dataset(
        {
            "precipitation": ("time", [1.0]*24)
        },
        {
            "time": np.arange(
                start_date,
                start_date + np.timedelta64(24, "h"),
                np.timedelta64(1, "h")
            )
        }                
    )
    return ds
    
def test_set_stationary_spg_output():
    ds = create_dataset()
    sc = StationaryConverter()
    sc.set_stationary_spg_output(ds)

def test_resample():
    ds = create_dataset()
    sc = StationaryConverter()
    sc.set_stationary_spg_output(ds)
    sc.resample_to_daily()
    assert len(sc.daily_ds) == 1
    
def date_to_tprime_converter(dates):
    return np.array([1]*len(dates))
    
def test_calc_delta_t_prime():
    ds = create_dataset()
    sc = StationaryConverter()
    sc.set_stationary_spg_output(ds)
    sc.resample_to_daily()
    tprime_training = 0.3
    sc.calc_delta_t_prime(date_to_tprime_converter, tprime_training)
    assert "delta_t_prime" in sc.daily_ds.keys()
    
def create_dataset_with_linearly_increasing_precipitation():
    # Create hourly dataset of 100x24 values from 0 to 99 repeated 24 times
    # (making the quantile calculation trivial)
    start_date = np.datetime64("2000-01-01")
    precip = np.array([[i/24]*24 for i in range(101)]).flatten()
    
    ds = xr.Dataset(
        {
            "precipitation": ("time", precip)
        },
        {
            "time": np.arange(
                start_date,
                start_date + np.timedelta64(len(precip), "h"),
                np.timedelta64(1, "h")
            )
        }                
    )
    return ds
    
def test_convert_precipitation_to_quantile():
    ds = create_dataset_with_linearly_increasing_precipitation()
    sc = StationaryConverter()
    sc.set_stationary_spg_output(ds)
    sc.resample_to_daily()
    tprime_training = 0.5
    sc.calc_delta_t_prime(date_to_tprime_converter, tprime_training)
    sc.convert_precipitation_to_quantile()
    
    assert "quantile" in sc.daily_ds.keys()
    epsilon = 1e-5
    assert np.all(sc.daily_ds["quantile"].values - range(101) < epsilon)

def interpolate_rate(quantiles):
    return np.array([0.1]*len(quantiles))
    
def test_calc_rate():
    ds = create_dataset_with_linearly_increasing_precipitation()
    sc = StationaryConverter()
    sc.set_stationary_spg_output(ds)
    sc.resample_to_daily()
    tprime_training = 0.5
    sc.calc_delta_t_prime(date_to_tprime_converter, tprime_training)
    sc.convert_precipitation_to_quantile()
    sc.calc_rate(interpolate_rate)
    
    assert "rate" in sc.daily_ds.keys()
    epsilon = 1e-5
    assert np.all(sc.daily_ds["rate"].values - 0.1 < epsilon)

def test_calc_multiplier():
    ds = create_dataset_with_linearly_increasing_precipitation()
    sc = StationaryConverter()
    sc.set_stationary_spg_output(ds)
    sc.resample_to_daily()
    tprime_training = 0.5
    sc.calc_delta_t_prime(date_to_tprime_converter, tprime_training)
    sc.convert_precipitation_to_quantile()
    sc.calc_rate(interpolate_rate)
    sc.calc_multiplier()
    
    assert "multiplier" in sc.daily_ds.keys()
    epsilon = 1e-5
    rate = 0.1 # from interpolate_rate
    expected = np.exp(tprime_training*rate)
    assert np.all(sc.daily_ds["multiplier"].values - expected < epsilon)
