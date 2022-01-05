import jax
from jax._src.api import named_call
from jax.scipy.optimize import minimize
import scipy.stats as ss
from functools import partial
import numpy as np

import jax.numpy as jnp
from jax import nn
from jax import grad
from jax import random
from jax.experimental.host_callback import id_print

import matplotlib.pyplot as plt

import tensorflow_probability as tfp
tfp = tfp.substrates.jax
tfd = tfp.distributions


# Important for convergence
jax.config.update("jax_enable_x64", True)


def lin_regress(x, y):
    return jnp.linalg.inv(x.T @ x) @ (x.T @ y)


def logistic_regress(x, w):
    return nn.sigmoid(x @ w)


def prob_bin(probs, thresh=0.5):
    """ Calculates binary predictions """
    pred_bin = jnp.array(probs)
    pred_bin = jnp.where(pred_bin > thresh, pred_bin, 0.0)
    pred_bin = jnp.where(pred_bin <= thresh, pred_bin, 1.0)
    return pred_bin


def logistic_loss(w, x, y, eps=1e-14, weight_decay=0.1):
    """ Generic nll with regularization from 
        https://www.architecture-performance.fr/ap_blog/logistic-regression-with-jax/"""
    #n = y.size
    p = logistic_regress(x, w)
    p = jnp.clip(p, eps, 1 - eps)  # bound the probabilities
    return -jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p)) + 0.5 * weight_decay * w@w


def nll_loss(preds, targets, eps=1e-14):
    p = jnp.clip(preds, eps, 1 - eps)
    y = jnp.clip(targets, eps, 1 - eps)
    return -jnp.sum(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))


def fit(func, params_init, options={"gtol": 1e-7}):
    return minimize(func, params_init, method="BFGS", options=options)


def fill_first(params, n=1, val=0):
    """ Adds a zero to params, so we can force the loc=0 """
    return jnp.concatenate([jnp.full(n, val), params], axis=None)


def fill_last(params, n=1, val=0):
    """ Adds a zero to params, so we can force the loc=0 """
    return jnp.concatenate([params, jnp.full(n, val)], axis=None)


class Dist():
    def __init__(self, name, params):
        self.name = name
        self.params = params
        if self.params is not None:
            print(f'Created distribution, {self.name} with {len(self.params)} elements')
        

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

    def __repr__(self, ):
        param_str = ', '.join([str(p) for p in self.params])
        return f'{self.name}, params: {param_str}'


class TFPDist(Dist):
    def __init__(self, dist, name, num_params=2, param_func=None, param_init=None):
        if param_init is None:
            param_init = [1.0]*num_params

        self.dist = dist
        self.param_func = param_func
        self._scale = None
        super().__init__(name, jnp.array(param_init))

    def fit(self, data, fit_func=fit, eps=1e-7):
        # self._scale = data.std()
        # data = data/self._scale

        def loss_func(params):
            if self.param_func:
                params = self.param_func(params)
            res = (-self.dist(*params).log_prob(data+eps)).mean()
            return res

        res = fit_func(loss_func, self.params)
        self.params = res.x

        #assert res.success


class TFWeibull(TFPDist):
    def __init__(self, param_init=None):
        if param_init is None:
            param_init = [0.75, 1.0]

        super().__init__(tfd.Weibull, 'TFWeibull', num_params=2, param_init=param_init)
        self.ss_dist = ss.weibull_min

    def ppf(self, x, cond=None):
        params = self.params
        if self.param_func:
            params = self.param_func(params)

        shape, scale = self.params
        return self.ss_dist.ppf(x, shape, loc=0, scale=scale)#*self._scale

    def fit(self, data):
        self.params = self.params.at[-1].set(data.std())
        super().fit(data)


class TFGeneralizedPareto(TFPDist):
    def __init__(self, param_init=None):
        if param_init is None:
            param_init = [1.0, 0.1]

        super().__init__(tfd.GeneralizedPareto, 'TFGenpareto', num_params=2,
                         param_init=param_init, param_func=fill_first)
        self.ss_dist = ss.genpareto

    def ppf(self, x, cond=None):
        
        params = self.params
        if self.param_func:
            params = self.param_func(params)

        loc, scale, shape = params
        return self.ss_dist.ppf(x, shape, scale=scale, loc=loc)#*self._scale


class SSDist(Dist):
    def __init__(self, ss_dist, name):
        self.dist = ss_dist
        params = None
        #self._scale = None
        super().__init__(name, params)

    def ppf(self, p, cond=None):
        assert self.params is not None
        assert p >= 0 and p <= 1.0

        return self.dist.ppf(p, *self.params)#*self._scale

    def fit(self, data, eps=1e-12):
        #self._scale = data.std()
        # data = data/self._scale
        self.params = self.dist.fit(data + eps, floc=0.0)


SSWeibull = partial(SSDist, ss.weibull_min, 'SSWeibull')
SSGeneralizedPareto = partial(SSDist, ss.genpareto, 'SSGenpareto')

class RainDay(Dist):
    def __init__(self, thresh=0.1, ar_depth=1, rnd_key=random.PRNGKey(42)):
        """ Calculates wether or not it is a rain day based on the last n dry/wet days """
        self.thresh = thresh
        self.ar_depth = ar_depth

        params = random.normal(rnd_key, shape=(ar_depth+1, ))
        self.did_fit = False
        super().__init__('RainDay', params)

    def _process_x(self, x, dim):
        concat = []
        if x is not None and len(x) > 0:
            x = jnp.array(x)
            concat.append(x)

        # Add the bias
        concat.append(jnp.ones((dim, 1)))

        return jnp.concatenate(concat, axis=1).astype(float)

    def get_thresh(self, x=None):
        x = self._process_x(x, dim=len(x))
        return logistic_regress(x, self.params)

    def ppf(self, p, x=None):
        assert x is None or self.ar_depth > 0

        p = jnp.atleast_1d(p)
        thresh = self.get_thresh(x)

        return p < thresh

    def fit(self, data, fit_func=fit):
        cols = []
        # TODO: This assumes that there are no gaps in the time series
        for n in range(self.ar_depth):
            cols.append(data[n: -(self.ar_depth-n)] > self.thresh)

        y = prob_bin(data[self.ar_depth:], self.thresh)
        x = self._process_x(jnp.stack(cols, axis=1), dim=len(y))

        def train_func(params):
            return logistic_loss(params, x, y)

        res = fit_func(train_func, self.params)
        #res = fit_func(train_func, params_init=self.params)
        self.params = res.x
        self.did_fit = res.success
