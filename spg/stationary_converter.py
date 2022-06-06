import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


class StationaryConverter:
    def __init__(self):
        pass

    def set_stationary_spg_output(self, dataset):
        assert "time" in dataset.keys()
        assert "precipitation" in dataset.keys()
        dataset = dataset.drop("tprime") # we don't use this
        self.hourly_ds = dataset
    
    def load_stationary_spg_output(self, path):
        print(f"Loading hourly stationary-SPG output from {path}")
        dataset = xr.open_dataset(path)
        self.set_stationary_spg_output(dataset)

    def resample_to_daily(self):
        print("Resampling hourly data to daily")
        self.daily_ds = self.hourly_ds.resample(time="D").sum()

    def calc_delta_t_prime(self, date_to_t_tprime_converter, t_prime_training):
        print("Calculating delta t_prime")
        convert_date_to_t_tprime = date_to_t_tprime_converter
        t_primes = convert_date_to_t_tprime(self.daily_ds["time"])
        self.daily_ds = self.daily_ds.assign(t_prime=("time",t_primes))
        delta_t_primes = self.daily_ds["t_prime"].values - t_prime_training
        self.daily_ds = self.daily_ds.assign(delta_t_prime=("time",delta_t_primes))

    def convert_precipitation_to_quantile(self):
        print("Converting precipitation to quantile")
        # Obtain the mapping from value to quantile
        sorted_precipitation = sorted(self.daily_ds["precipitation"].to_numpy())
        quantiles = np.linspace(0,1, num=len(sorted_precipitation))
        precipitation_to_quantile = interp1d(sorted_precipitation, quantiles)

        import warnings
        # For zero-valued precipitation a warning is generated; this can be safely ignored
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quantiles_for_precipitation = precipitation_to_quantile(self.daily_ds["precipitation"])
        quantiles_for_precipitation[np.isnan(quantiles_for_precipitation)] = 0

        self.daily_ds = self.daily_ds.assign(quantile=("time", quantiles_for_precipitation))

    def calc_rate(self, interpolate_rate):
        print("Calculating rates")
        quantiles = self.daily_ds["quantile"].to_numpy()
        rates = interpolate_rate(quantiles)
        self.daily_ds = self.daily_ds.assign(rate=("time",rates))

    def calc_multiplier(self):
        multipliers = np.exp(
            self.daily_ds["delta_t_prime"].to_numpy() 
            * self.daily_ds["rate"].to_numpy()
        )
        self.daily_ds = self.daily_ds.assign(multiplier=("time", multipliers))

    def calc_daily_non_stationary_precipitation(self):
        non_stationary_precipitation = (
            self.daily_ds["precipitation"].to_numpy() 
            * self.daily_ds["multiplier"].to_numpy()
        )
        self.daily_ds = self.daily_ds.assign(
            non_stationary_precipitation=("time",non_stationary_precipitation)
        )

    def save_daily(self, filepath):
        print(f"Saving {filepath}")
        self.daily_ds.to_netcdf(filepath)

    def calc_hourly_non_stationary_precipitation(self, date_to_t_tprime_converter):
        print("Calculating hourly non-stationary precipitation from daily")
        # Get date from hourly timestamps
        timestamps = self.hourly_ds["time"].to_numpy()
        dates = timestamps.astype('datetime64[D]')

        # Get multipliers for the given dates
        multipliers = self.daily_ds["multiplier"].sel(time=dates).to_numpy()
        self.hourly_ds = self.hourly_ds.assign(multipler=("time", multipliers))

        # Apply multipliers to generate non-stationary precipitation
        non_stationary_precipitation = self.hourly_ds["precipitation"].to_numpy() * multipliers
        self.hourly_ds = self.hourly_ds.assign(
            non_stationary_precipitation=("time", non_stationary_precipitation)
        )

        t_primes = date_to_t_tprime_converter(dates)
        self.hourly_ds = self.hourly_ds.assign(
            t_prime=("time", t_primes)
        )
        
        
    def save_hourly(self, filepath):
        print(f"Saving {filepath}")
        self.hourly_ds.to_netcdf(filepath)

        
