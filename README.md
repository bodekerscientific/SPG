# SPG

A single-site, hourly and daily deep learning based, stochastic precipitation generator.

See the [GMD paper](https://gmd.copernicus.org/articles/16/3785/2023/)

### Installation

First install Anaconda (or even better Mamba), the SPG has only been tested on Linux. Then the environment can be created with:

    conda create --file environment.yml --name spg
    conda activate spg

You can install the spg as a package with:

    python -m pip install --editable .

### Configuration

Prepare the precipitation data as NetCDF files, either at daily or hourly frequency. The code expects each NetCDF file to be named after a single site, with a single coordinate - time and a variable named precipitation.

Update the config files under base_daily or base_daily, add your new location and update the input and output paths. Optional setup weights and biases for tracking runs.

Choose the version you want to use, under config/modeles. v10 is recommended. Be sure to include the following if you want to train a stationary SPG (including a post-hoc correction version):

    loader_args :
      inc_tprime : FALSE

v8 is the non-stationary version of v10, however cation should be used when training a non-stationary SPG on observations alone.

### Training

Training if spg with:

    python spg/train_spg.py location config_base version

For example:

    python spg/train_spg.py auckland base_daily v10

After training, review the validation qq plots under output_path and select an epoch that you want.

### Generation of synthetic precipitation series

You can now produce synthetic precipitation series using spg/run.py with your desired epoch:

    python spg/run.py location

For example:

    python spg/run.py auckland v10 9 base_daily

The number of ensemble members, and the SSPs/RCPs to be used, are configured in the base config. By default, one run is produced over the historical time range, then an ensemble for each of the scenarios are produced from 1980, to 2100. If the stationary SPG is used, then the SSP/year will have no impact on the results.

### Applying a post-hoc correction

After producing an ensemble of non-stationary simulations, notebooks/make-many-non-stationary.ipynb applies the post-hoc correction, however climate model simulations are needed to calculate the correction coefficients.
