import scipy.stats as ss
from functools import partial
import numpy as np

import jax.numpy as jnp
from jax import nn
from jax import grad
from jax import random


import matplotlib.pyplot as plt
from jax.scipy.optimize import minimize


def lin_regress(x, y):
    return jnp.linalg.inv(x.T @ x) @ (x.T @ y)


def logistic_regress(x, w):
    return nn.sigmoid(x @ w)

def prob_bin(probs, thresh=0.5):
    """ Calculates binary predictions """
    pred_bin = jnp.array(probs)
    pred_bin = jnp.where(pred_bin < thresh, pred_bin, 1.0)
    pred_bin = jnp.where(pred_bin >= thresh, pred_bin, 0.0)
    return pred_bin


def logistic_loss(w, x, y, eps=1e-14, weight_decay=0.1):
    """ Generic nll with regularization from 
        https://www.architecture-performance.fr/ap_blog/logistic-regression-with-jax/"""
    #n = y.size
    p = logistic_regress(x, w)
    p = jnp.clip(p, eps, 1 - eps)  # bound the probabilities
    return -jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))+ 0.5 * weight_decay * w@w


def nll_loss(preds, targets, eps=1e-14):
    p = jnp.clip(preds, eps, 1 - eps)
    y = jnp.clip(targets, eps, 1 - eps)
    return -jnp.sum(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))


def fit(func, params_init, options={"gtol": 1e-5}):
    return minimize(func, params_init, method="BFGS", options=options)
    
class Dist():
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def cdf(self, x, cond=None):
        raise NotImplementedError('Need to make a subclass')

    def ppf(self, p, cond=None):
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

    def ppf(self, p, cond=None):
        assert self.params is not None
        assert p >=0 and p <= 1.0
        
        return self.dist.ppf(p, *self.params)

    def fit(self, data):
        self.params = self.dist.fit(data)
    
class RainDay(Dist):
    def __init__(self, thresh=0.1, ar_depth=1, rnd_key=random.PRNGKey(42)):
        """ Calculates wether or not it is a rain day based on the last n dry/wet days """
        super().__init__(self)
        self.thresh = thresh
        self.ar_depth = ar_depth

        self.params = random.normal(rnd_key, shape=(ar_depth+1, ))
        self.did_fit = False

        print(f'Created model with {len(self.params)} elements')

    def _process_x(self, x, dim):
        concat = []
        if x is not None and len(x) > 0:
            if isinstance(x, list):
                x = jnp.stack(x, axis=1)
            concat.append(x)

        # Add the bias        
        concat.append(jnp.ones((dim, 1)))

        return jnp.concatenate(concat, axis=1).astype(float)

    def ppf(self, p, x=None):
        assert x is None or self.ar_depth > 0

        #assert len(x) == self.ar_depth
        
        p = jnp.atleast_1d(p)
        x = self._process_x(x, dim=len(p))

        return prob_bin(p, thresh=logistic_regress(x, self.params))#.astype(bool)

    def fit(self, data, fit_func=fit):
        cols = []
        # TODO: This assumes that there are no gaps in the time series
        for n in range(self.ar_depth):
            cols.append(data[n: -(self.ar_depth-n)] > self.thresh)
        
        y = prob_bin(data[self.ar_depth:], self.thresh)
        x = self._process_x(cols, dim=len(y))

        def train_func(params):
            return logistic_loss(params, x, y)

        res = fit_func(train_func, self.params)
        #res = fit_func(train_func, params_init=self.params)
        self.params = res.x
        self.did_fit = res.success
        

Weibull = partial(SSDist, ss.weibull_min)
GPD = partial(SSDist, ss.genpareto)
