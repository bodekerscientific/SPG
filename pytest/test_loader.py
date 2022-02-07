from spg import data_loader, data_utils
import numpy as np
import pandas as pd
import cftime

def get_ds(data=None):
    if data is None:
        data = data_utils.load_data()

    return data_loader.PrecipitationDataset(data)

def test_dataset_hourly():
    data = data_utils.load_data_hourly()
    ds = get_ds(data)

    print(len(ds))

def test_get_daily_features():
    precip = np.array([10, 1.0, 0, 0.05 , 20])
    dts = pd.date_range(start=pd.Timestamp('2019-01-01'), end=pd.Timestamp('2019-01-05'))
    df = pd.Series(precip, index=dts)
    x, y = data_loader.generate_features_daily(df, average_days=[1, 4])
    
    assert x.shape[0] == 1
    x = x[0]

    tprime = data_utils.load_magic()
    tprime = tprime['ssp245'][tprime.index.year == 2019].values
    
    assert np.isclose(tprime, x[-1])
    assert precip[-1] == y[0]
    assert (x[0:2] >= -1).all()
    assert (x[0:2] <= 1).all()

    assert np.isclose(x[2], precip[-2])
    assert x[3] == 0
    assert np.isclose(x[4], precip[0:-1].mean())
    assert np.isclose(x[5], 0.5)


def test_dts_360():
    dts = pd.date_range(start=pd.Timestamp('2019-01-01'), end=pd.Timestamp('2019-01-05'))
    dts = np.array([cftime.Datetime360Day(dt.year, dt.month, dt.day) for dt in dts])
    target = 2019 + np.linspace(1/360, len(dts)/360, len(dts), endpoint=True)
    
    assert np.isclose(target, data_loader.dt_360_to_dec_year(dts)).all()

def test_get_daily_features_wh():
    precip = np.array([10, 1.0, 0, 0.05 , 20])
    
    dts = pd.date_range(start=pd.Timestamp('2019-01-01'), end=pd.Timestamp('2019-01-05'))
    dts = np.array([cftime.Datetime360Day(dt.year, dt.month, dt.day) for dt in dts])
    print(dts)
    
    tp = 1.0
    df = pd.DataFrame({'pr' : precip, 'dts' :dts, 'tp' : tp})
    
    x, y = data_loader.generate_features_daily(df, average_days=[1, 4], is_wh=True)
    
    assert x.shape[0] == 1
    x = x[0]
    
    assert np.isclose(tp, x[-1])
    assert precip[-1] == y[0]
    assert (x[0:2] >= -1).all()
    assert (x[0:2] <= 1).all()

    assert np.isclose(x[2], precip[-2])
    assert x[3] == 0
    assert np.isclose(x[4], precip[0:-1].mean())
    assert np.isclose(x[5], 0.5)

def test_wh_loader():
    wh_dfs = data_utils.load_wh(num_ens=3)
    wh_tr, wh_val = data_loader.get_datasets(wh_dfs, num_valid=1, is_wh=True)

    assert len(wh_tr) == 360*2*3
    assert len(wh_val) == 360*3


if __name__ == '__main__':
    test_wh_loader()