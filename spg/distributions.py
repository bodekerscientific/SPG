import scipy.stats as ss
from functools import partial
import jax.numpy as jnp
import numpy as np
from jax import nn

def lin_regress(x, y):
    return jnp.linalg.inv(x.T @ x) @ (x.T @ y)

def logistic_regress(x, w, c):
    return nn.sigmoid(x@w + c)

def logistic_loss(c, w, x, y, eps=1e-14, weight_decay=0.1):
    n = y.size
    p = logistic_regress(c, w, x)
    p = jnp.clip(p, eps, 1 - eps)  # bound the probabilities within (0,1) to avoid ln(0)
    return -jnp.sum(y * jnp.log(p) + (1 - y) * jnp.log(1 - p)) / n + 0.5 * weight_decay * (
        jnp.dot(w, w) + c * c
    )


class Dist():
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def cdf(self, x):
        raise NotImplementedError('Need to make a subclass')

    def fit(self, data):
        raise NotImplementedError('Need to make a subclass')

    def eval_tr(self, params):
        for func in self.transforms:
            params = func(params)
        return params

class SSDist(Dist):
    def __init__(self, ss_dist, transforms=[]):
        super().__init__(transforms)
        self.dist = ss_dist
        self.params = None

    def cdf(self, x):
        assert self.params is not None
        x = self.eval_tr(x)

    def fit(self, data):
        x = self.eval_tr(x)

    def eval_dist(self, params):
         raise NotImplementedError('Need to make a subclass')


class RainDay(Dist):
    def __init__(self, thresh=0.1, ar_depth=1):
        """ Calculates wether or not it is a rain day based on the last n dry/wet days """
        super().__init__(self)
        self.thresh = thresh
        self.ar_depth = ar_depth

        self.coefs = jnp.zeros(ar_depth) + 1e-5
        self.bias = jnp.zeros(1) + 1e-5
        print(f'Created a transition table with {self.coef} elements')

    def ppf(self, x):
        assert np.isfinite(self.prob_table).all()

        if self.ar_depth > 0:
            assert len(x) == self.ar_depth + 1, print(f'Need to supply {len(self.ar_depth + 1)} data points for this ar model')
            is_wet = x[:-1] >= self.thresh

            return 0.0 if self.prob_table[is_wet] > x[-1] else 1.0  
        else:
            return 0.0 if self.prob_table > x else 1.0  

    def fit(self, data):
        cols = []
        # TODO: This assumes that there are no gaps in the time series
        for n in range(self.ar_depth):
            cols.append(data[n: len(data)-n] > data)
        cols.append(jnp.ones(len(data) - self.ar_depth))

        x = jnp.stack(cols, dim=1).astype(jnp.float64)        
        y = data

        # assert np.isfinite(self.prob_table).all(), 'Fit failed, try including reducing the ar depth'
        # print(f'Minimum count {total.min()}')

Weibull = partial(SSDist, ss.weibull_min)
GPD = partial(SSDist, ss.genpareto)
