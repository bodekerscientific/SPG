from spg import distributions
import scipy.stats as ss
import numpy as np
from jax.scipy.optimize import minimize
from jax import grad
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
# import tensorflow_probability as tfp; tfp = tfp.substrates.jax
# tfd = tfp.distributions


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


def get_data_rnd(n=1000):
    a = np.random.normal(scale=1.0, size=n+1)**4
    return (a[1:] + a[:-1])**0.1


def test_prob_bin():
    prob_bin = jnp.array([0.1, 0.51, 0.78, 0.99, 0])
    target = jnp.array([0, 1, 1, 1, 0])
    assert (distributions.prob_bin(prob_bin) == target).all()

def test_dry_logistic_cost():
    dataset = get_sklearn_test_data()
    w_0 = 1.0e-5 * jnp.ones(dataset['n_feat'])
    #print(grad(distributions.logistic_loss, argnums=1)(c_0, w_0, dataset['x_train'], dataset['y_train']))
    
    def fun(coefs):
        return distributions.logistic_loss(coefs, dataset['x_train'], dataset['y_train'])

    res = minimize(
        fun,
        w_0,
        method="BFGS",
        options={"gtol": 1e-5},
    )
    #assert res.success

    y_pred = distributions.prob_bin(distributions.logistic_regress(dataset['x_test'], res.x))
    assert accuracy_score(dataset['y_test'], y_pred) > 0.94


def test_dry_day():
    dist = distributions.RainDay(thresh=0.5, ar_depth=0)
    data = get_data()
    dist.fit(data)
    assert np.isclose(dist.params[0], 0, atol=1e-4)
    assert dist.ppf(0.7) == 1.0
    assert dist.ppf(0.499) == 0

def test_dry_day_ar():
    dist = distributions.RainDay(thresh=1.0, ar_depth=2)
    data = jnp.array(get_data_rnd())
    dist.fit(data)
    assert len(dist.params) == 3
    print(dist.params)
    print(dist.ppf(0.6, x=jnp.array([[1, 1]])))
    print(dist.ppf(0.7, x=jnp.array([[0, 1]])))
    print(dist.ppf(0.7, x=jnp.array([[0, 0]])))

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

if __name__ == '__main__':
    test_tf_weibull()