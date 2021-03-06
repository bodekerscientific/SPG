from spg import data_loader, data_utils
import numpy as np
import pandas as pd
import cftime

def get_ds(data=None):
    if data is None:
        data = data_utils.load_data()

    return data_loader.PrecipitationDataset(data)



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

def test_feat_multiscale():
    precip = np.array([np.nan, 10, 1.0, 0, 0, 0, 0.05 , 20, 11, 1, 0, np.nan])
    
    exp_ratio = np.array([1.0, 0.0, 0.05/(20+0.05), 20/(20+11), 11/12, 1, 1.0, 0.0, 0.0, 0.05/(20+0.05+11), 20/(20+11+1), 11/12])
    exp_x = np.array([10, 0, 0, 0.05, 20, 11, 10, 0, 0, 0, 0.05, 20,])


    dts = pd.date_range(start=pd.Timestamp('2019-01-01'), periods=len(precip), freq='H')
    pr = pd.Series(precip, index=dts)

    res = data_loader.generate_features_multiscale(pr, max_hrs=3, cond_hr=1)
    
    for v in res.values():
        assert v.shape[0] == exp_x.shape[0]

    assert (res['ratio'] == exp_ratio).all()
    assert (res['x'][:, 0] == exp_x).all()

    
    
def test_generate_features_split():
    precip = np.array([10, 1.0, 0, 0, 0, 1.0 , 20, 11, 1, 0])
    dts = pd.date_range(start=pd.Timestamp('2019-01-01'), periods=len(precip), freq='H')
    pr = pd.Series(precip, index=dts)
    
    target = {
        'ratio' : np.array([[0., 0., 1.0], [0.0, 1/21, 20/21]]),
        'x' : np.array([[11.0, 1.0, 32.0, 1.0, 0,], [1.0, 21, 12, 0, 0,]]),
        'pr' : np.array([1.0, 21])
    }
    
    res = data_loader.generate_features_split(pr, sum_period=3, cond_hr=2)
    for k,v in res.items():
        assert np.isclose(v, target[k], 1e-4, 1e-4).all()
    
    

if __name__ == '__main__':
    test_generate_features_split()