from spg.generator import SPG
from spg import distributions 

import pandas as pd
from jax import random

def load_data(fpath):
    df = pd.read_csv(fpath, sep='\t', parse_dates=['Date(UTC)'], skiprows=8)

    df['date'] = pd.to_datetime(df['Date(UTC)'].values, format='%Y%m%d:%H%M')
    return pd.Series(df['Amount(mm)'].values, index=df['date'].values)


def fit_and_generate(data,):
    rd = distributions.RainDay(thresh=1.0, ar_depth=2)
    rain_dists = { 0 : distributions.Weibull(),
                  0.995 : distributions.GPD()}
    rng = random.PRNGKey(42)

    sp = SPG(rd, rain_dists, rng)
    sp.fit(data.values)

    

if __name__ == '__main__':
    fpath = "/mnt/temp/projects/emergence/data_keep/station_data/dunedin_btl_gardens_precip.tsv"
    data = load_data(fpath)
    fit_and_generate(data)

