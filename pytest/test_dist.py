"""
    @Author Leroy Bird
    Tests for spg.distributions
"""

import scipy.stats as ss
import numpy as np
from jax.scipy.optimize import minimize
import jax.numpy as jnp
import jax
from functools import cache
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
# import tensorflow_probability as tfp; tfp = tfp.substrates.jax
# tfd = tfp.distributions

from spg import distributions, jax_utils
from spg import run
from spg.generator import cycle

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_data():
    return np.array([1.0, 10.0, 8, 7, 0, 0, 0, 0, 17, 0.12]*20)


def get_sklearn_test_data():
    # From https://www.architecture-performance.fr/ap_blog/logistic-regression-with-jax/
    X, y = load_breast_cancer(return_X_y=True)

    # Add ones to the dataset which creates a 'bias'
    
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=142
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    n_feat = X.shape[1]

    return {'x_train': scaler.transform(X_train),
            'x_test': scaler.transform(X_test),
            'y_train': y_train,
            'y_test': y_test,
            'n_feat' : n_feat}
            
@cache
def get_real_data():
    fpath = "/mnt/temp/projects/emergence/data_keep/station_data/dunedin_btl_gardens_precip.tsv"
    return run.load_data(fpath)


def get_data_rnd(n=1000):
    np.random.seed(42)
    a = np.random.normal(scale=1.0, size=n+1)**4
    return (a[1:] + a[:-1])

def get_data_rnd_non_stationary(n=1000):
    a = get_data_rnd(n)
    t = np.linspace(0, 1.0, n)
    a = a*(t*0.5 + 1.0)
    return t, a

def test_pos_only():
    assert (jax_utils.pos_only(jnp.array([-10., -1, 3])) > 0).all()
    assert jax_utils.pos_only(jnp.array([-10., -1, 3]))[-1] == 4
    a = jnp.array([-10.0, -1.0, 3.0])
    b = jax_utils.apply_pos(a, 0)
    assert (a[1:] == b[1:]).all()
    assert b[0] >= 0
    assert jnp.isclose(jax_utils.pos_only(-1000000), 0)



def test_prob_bin():
    prob_bin = jnp.array([0.1, 0.51, 0.78, 0.99, 0])
    target = jnp.array([0, 1, 1, 1, 0])
    assert (jax_utils.prob_bin(prob_bin) == target).all()

def test_dry_logistic_cost():
    dataset = get_sklearn_test_data()
    w_0 = 1.0e-5 * jnp.ones(dataset['n_feat'])
    #print(grad(distributions.logistic_loss, argnums=1)(c_0, w_0, dataset['x_train'], dataset['y_train']))
    
    def fun(coefs):
        return jax_utils.logistic_loss(coefs, dataset['x_train'], dataset['y_train'])

    res = minimize(
        fun,
        w_0,
        method="BFGS",
        options={"gtol": 1e-5},
    )
    #assert res.success

    y_pred = jax_utils.prob_bin(jax_utils.logistic_regress(dataset['x_test'], res.x))
    assert accuracy_score(dataset['y_test'], y_pred) > 0.94


def test_dry_day_ar():
    dist = distributions.RainDay(thresh=1.0, ar_depth=2)
    data = jnp.array(get_data_rnd())
    dist.fit(data)
    assert len(dist.get_params()) == 3
    print(dist.get_params)
    print(dist.ppf(0.6, x=jnp.array([[1, 1]])))
    print(dist.ppf(0.7, x=jnp.array([[0, 1]])))
    print(dist.ppf(0.7, x=jnp.array([[0, 0]])))


def test_dry_day_ar_order(data = get_real_data()):
    dist = distributions.RainDay(thresh=0.1, ar_depth=2)
    data = jnp.array(data)
    dist.fit(data)
    assert len(dist.get_params()) == 3
    
    print(dist.get_params())
    a = dist.get_thresh(x=jnp.array([[1, 1]]))
    b = dist.get_thresh(x=jnp.array([[0, 0]]))
    print(a, b)
    assert a > b

    rain_days = []
    np.random.seed(42)
    rnd_vec = np.random.uniform(0, 1.0, size=1000)
    last = jnp.array([[1.0, 1.0]])
    
    for rnd in rnd_vec:
        a = dist.ppf(rnd, x=last)
        last = cycle(last, a.astype(float))
        rain_days.append(a[0])

    rain_days = np.array(rain_days).astype(float)

    assert np.isclose(rain_days.mean(), float((data >= .1).sum()/data.size), rtol=1e-2, atol=1e-2) 

def test_tf_gpd():
    loc, scale, shape = (.3, 1.1, 2.1)
    dist = tfd.GeneralizedPareto(scale=scale, loc=loc, concentration=shape)
    assert np.isclose(dist.cdf(0.5), ss.genpareto.cdf(0.5, shape, loc=loc, scale=scale))

def test_tf_weibull():
    scale, shape = (1.1, 2.1)
    dist = tfd.Weibull(scale=scale, concentration=shape)
    
    val = dist.cdf(0.5)    
    target = ss.weibull_min.cdf(0.5, shape, loc=0, scale=scale)

    assert np.isclose(val, target)

def test_tf_weibull_fit(data = get_data_rnd()):    
    data = data[data>=.1] - .1
    #data[data > 50] = 0.0
    #data /= data.std()
    tf_dist = distributions.TFWeibull()
    ss_dist =  ss.weibull_min

    coefs_ss = list(ss_dist.fit(data, floc=0))
    tf_dist.fit(data)
    print(len(data))

    print((-np.log(ss_dist.pdf(data, *coefs_ss))).mean())
    print(coefs_ss)
    print(tf_dist.get_params())

    # Delete location param
    del coefs_ss[1]
    assert(np.isclose(tf_dist.get_params(), coefs_ss, atol=1e-4, rtol=1e-3).all())
    assert np.isclose(tf_dist.ppf(.9), ss_dist.ppf(0.9, c=coefs_ss[0], scale=coefs_ss[1]))

def test_tf_gpd_fit(data=get_data_rnd(100_000)):
    thresh = np.quantile(data, 0.99)
    data = data[data >= thresh] - thresh
    #data /= data.std()

    tf_dist = distributions.TFGeneralizedPareto()
    ss_dist =  ss.genpareto

    coefs_ss = list(ss_dist.fit(data, floc=0))
    coefs_ss = [coefs_ss[1], coefs_ss[-1], coefs_ss[0]]

    tf_dist.fit(data)
    print(f'ss : {coefs_ss} , tf : {tf_dist.get_params()}')
    assert(np.isclose(tf_dist.get_params(), coefs_ss, atol=1e-4, rtol=1e-3).all())
    #assert np.isclose(tf_dist.ppf(.9), ss_dist.ppf(0.9, shape, loc=loc, scale=scale), atol=1e-5, rtol=1e-4)

def test_real_data():
    data = get_real_data()
    test_tf_gpd_fit(data)
    test_tf_weibull_fit(data/data.std())

def _make_ns_plot(data, dist, output_path):
    plt.figure(figsize=(20, 14))
    x = jnp.linspace(0, 1, 1000, endpoint=False)
    t = jnp.ones_like(x)
    for tp in [0, 0.5, 1.0]:
        y = dist.ppf(x, cond={'tprime' : t*tp})
        plt.plot(x, y, label=tp)
    plt.plot(x, np.quantile(data, x), label='data')
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()

def test_tf_weibull_ns_fit(data=None, tprime=None):
    if data is None:
        tprime, data = get_data_rnd_non_stationary()

    mask = data>=.1
    data = data[mask] - .1
    tprime = tprime[mask]

    tf_dist = distributions.TFWeibull(param_init=[0.75, 1.0, 0.01, 0.01],
                                      param_func=jax_utils.linear_exp_split)
    ss_dist = ss.weibull_min

    coefs_ss = list(ss_dist.fit(data, floc=0))
    tf_dist.fit(data, cond={'tprime' : tprime})
    print(tf_dist)

    _make_ns_plot(data, tf_dist, 'non_stationary_weibull.png')

def test_tf_gpd_ns_fit(data=None, tprime=None):
    if data is None:
        data = get_real_data()
        tprime = np.linspace(0, 1.0, len(data))

    thresh = np.quantile(data, 0.99)
    mask = data > thresh
    data = data[mask] - thresh
    print(data)
    tprime = tprime[mask]

    tf_dist = distributions.TFGeneralizedPareto(param_init=[-1.0, 1.1, 0.01, 0.01],
                                                param_func=jax_utils.linear_exp_split)
    tf_dist.fit(data, cond={'tprime' : tprime})
    print(tf_dist)

    _make_ns_plot(data, tf_dist, 'non_stationary_gpd.png')

if __name__ == '__main__':
    test_real_data()