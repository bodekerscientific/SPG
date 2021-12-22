from spg import distributions
import scipy.stats as ss
import numpy as np

def get_data():
    return np.array([1.0, 10.0, 8, 7, 0, 0, 0, 0, 17, 0.12])

def get_data_rnd(n=1000):
    a =  np.random.normal(scale=2.0, size=n+1)**4
    return a[1:] + a[:-1]


def test_dry_day():
    dist = distributions.RainDay(thresh=0.5, ar_depth=0)
    data = get_data()
    dist.fit(data)
    assert dist.prob_table == 0.5
    assert dist.ppf(0.7) == 1.0
    assert dist.ppf(0.499) == 0

def test_dry_day_ar():
    dist = distributions.RainDay(thresh=0.5, ar_depth=2)
    data = get_data_rnd()
    dist.fit(data)
    print(dist.prob_table)

if __name__ == '__main__':
    test_dry_day_ar()
