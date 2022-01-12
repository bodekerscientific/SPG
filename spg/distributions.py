"""
    @Author Leroy Bird
    Distributions used in the SPG
"""

import jax
from jax._src.api import named_call
from jax.scipy.optimize import minimize
import scipy.stats as ss
from functools import partial
import numpy as np

import jax.numpy as jnp
import flax.linen as nn
from jax import random
from jax.experimental.host_callback import id_print

import matplotlib.pyplot as plt

from . import jax_utils

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

# Important for convergence
jax.config.update("jax_enable_x64", True)

def fit(func, params_init, options={"gtol": 1e-7}):
    return minimize(func, params_init, method="BFGS", options=options)


def fill_first(params, n=1, val=0):
    """ Adds a zero to params, so we can force the loc=0 """
    val = jnp.full((n,) + params.shape[1:] , val)
    return jnp.concatenate([val, params], axis=0)


def fill_last(params, n=1, val=0):
    """ Adds a zero to params, so we can force the loc=0 """
    val = jnp.full((n,) + params.shape[1:] , val)
    return jnp.concatenate([params, val], axis=0)


class Dist():
    """ Abstract distribution class which all the distributions must subclass """
    def __init__(self, name, params, param_post=None, param_func=None,):
        self.name = name
        self._params = params

        self.param_func = param_func
        self.param_post = param_post
        if self.param_func is None:
            self.param_func = lambda x, cond : x
        if self.param_post is None:
            self.param_post = lambda x : x
        
        self.offset = None
        self.max_prob = None
        if self._params is not None:
            print(f'Created distribution, {self.name} with {len(self._params)} elements')
        
    def get_params(self, cond=None):
        return self.param_post(self.param_func(self._params, cond))

    def cdf(self, x, cond=None):
        raise NotImplementedError('Need to make a subclass')

    def ppf(self, p, cond=None):
        raise NotImplementedError('Need to make a subclass')

    def fit(self, data, cond=None):
        raise NotImplementedError('Need to make a subclass')
        
    def __repr__(self, ):
        param_str = ', '.join([str(p) for p in self._params])
        return f'{self.name}, offset: {self.offset}, max_prob: {self.max_prob}, params: {param_str}'


class TFPDist(Dist):
    def __init__(self, dist, ss_dist, name, num_params=2, param_init=None, **kwargs):
        if param_init is None:
            param_init = [1.0]*num_params

        self.dist = dist
        self.ss_dist = ss_dist
        super().__init__(name, jnp.array(param_init), **kwargs)

    def fit(self, data, cond=None, fit_func=fit, eps=1e-12):
        def loss_func(params):
            params = self.param_post(self.param_func(params, cond=cond))
            res = (-self.dist(*params).log_prob(data+eps)).mean()
            #id_print(res)
            #id_print(params)
            return res

        res = fit_func(loss_func, self._params)
        self._params = res.x

    def get_ss_params(self, cond=None):
        raise NotImplementedError('Need to override this method')

    def ppf(self, x, cond=None):
        shape, loc, scale = self.get_ss_params(cond=cond)
        return self.ss_dist.ppf(x, shape, loc=loc, scale=scale)

    def cdf(self, x, cond=None):
        shape, loc, scale = self.get_ss_params(cond=cond)
        return self.ss_dist.cdf(x, shape, loc=loc, scale=scale)


class TFWeibull(TFPDist):
    def __init__(self, param_init=None, **kwargs):
        if param_init is None:
            param_init = [0.75, 1.0]

        # Ensure the scale and shape params are always positive
        def param_post(params):
            return jax_utils.pos_only(params)
        
        super().__init__(tfd.Weibull, ss.weibull_min, 'TFWeibull', num_params=2, 
                         param_init=param_init, param_post=param_post, **kwargs)

    def get_ss_params(self, cond=None):
        params = self.get_params(cond)
        return params[0], 0.0, params[1]


class TFGeneralizedPareto(TFPDist):
    def __init__(self, param_init=None, **kwargs):
        if param_init is None:
            param_init = [1.0, 0.2]

        def post_process(params):
            # Add back location first
            params = fill_first(params)

            # Ensure the scale param is always positive
            return jax_utils.apply_func_idx(params, idx=1)

        super().__init__(tfd.GeneralizedPareto, ss.genpareto, 'TFGenpareto', num_params=2,
                         param_init=param_init, param_post=post_process, **kwargs)

    def get_ss_params(self, cond=None):
        params = self.get_params(cond)
        return params[2], params[0], params[1]


class TFMixture(Dist):
    def __init__(self, dist_mix, num_mix, name='Mixture', param_init=None, num_comp=2, wd=0.03, rng=None, **kwargs):
        
        if rng is None:
            rng = jax.random.PRNGKey(42)
        self.rng = rng

        if param_init is None:
            params = jax.random.uniform(rng, (num_mix*(num_comp+1),)) + 0.2
        else:
            params = param_init
            assert len(params) == num_mix*(num_comp+1)

        self.wd = wd
        self.distComp = dist_mix
        super().__init__(name, params, **kwargs)

    def make_dist(self, params=None, cond=None):
        if params is None:
            params = self._params

        params, probs = self.param_func(params, cond)
        
        dist_mix = self.distComp(*params)
        return tfd.MixtureSameFamily(components_distribution=dist_mix, mixture_distribution=tfd.Categorical(probs=probs))

    def fit(self, data, cond=None, fit_func=fit, eps=1e-12, weighting=None):
        def loss_func(params, weight_decay=self.wd):
            res = self.log_prob(data, cond=cond, params=params, weighting=weighting)
            # id_print(res)
            # id_print(params)
            # We don't include the probs in the loss function
            wd =  0.5 * weight_decay * params[:-1]@params[:-1]
            #id_print(wd)
            return res + wd

        res = fit_func(loss_func, self._params)
        self._params = res.x
        assert res.success

    def log_prob(self, data, cond=None, params=None, eps=1e-12, weighting=None):
        if params is None:
            params = self._params
        
        if weighting is None:
            weighting = jnp.ones_like(data.size)
        else:
            weighting = jnp.array(weighting)

        return  -(self.make_dist(params, cond=cond).log_prob(data[:, None]+eps)*weighting).mean()

    def cdf(self, x, cond=None):
        return self.make_dist(cond=cond).cdf(x)

    def sample(self, n, cond=None, rng=None):
        if rng is None:
            rng = self.rng
        return self.make_dist(cond=cond).sample(n, seed=rng)

def post_process_mix(params, cond=None, num_comp=2, pos_only=True):
    """ Extract the paramaters for the mixture and the weighting of the params. """
    params = params.reshape((num_comp+1, -1))
    if pos_only:
        params = jax_utils.pos_only(params)
    
    probs = params[-1:]
    return params[:-1], nn.softmax(probs)

TFGammaMix = partial(TFMixture, dist_mix=tfd.Gamma, name='GammaMix', 
                    param_func=post_process_mix)

TFLogNormMix = partial(TFMixture, dist_mix=tfd.LogNormal, name='LogNormMix', 
                       param_func=post_process_mix)
 
TFWeibullMix = partial(TFMixture, dist_mix=tfd.Weibull, name='WeibullMix', 
                       param_func=post_process_mix)

TFInverseGammaMix = partial(TFMixture, dist_mix=tfd.InverseGamma, name='InverseGamma', 
                       param_func=post_process_mix)

class SSDist(Dist):
    def __init__(self, ss_dist, name):
        self.dist = ss_dist
        params = None
        super().__init__(name, params)
    
    def cdf(self, x, cond=None):
        assert self._params is not None
        return self.dist.cdf(x, *self.get_params(cond))

    def ppf(self, p, cond=None):
        assert self._params is not None
        assert p >= 0 and p <= 1.0

        return self.dist.ppf(p, *self.get_params(cond))

    def fit(self, data, cond=None, eps=1e-12):
        self._params = self.dist.fit(data + eps, floc=0.0)


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
        return jax_utils.logistic_regress(x, self._params)

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

        y = jax_utils.prob_bin(data[self.ar_depth:], self.thresh)
        x = self._process_x(jnp.stack(cols, axis=1), dim=len(y))

        def train_func(params):
            return jax_utils.logistic_loss(params, x, y)

        res = fit_func(train_func, self._params)
        #res = fit_func(train_func, params_init=self.params)
        self._params = res.x
        self.did_fit = res.success
