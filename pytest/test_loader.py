from spg import data_loader, data_utils
import numpy as np
import pandas as pd

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

if __name__ == '__main__':
    test_get_daily_features()