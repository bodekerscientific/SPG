
from typing import Iterable, Sequence, Callable, Any
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from functools import partial

from spg import jax_utils

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

jax.config.update("jax_debug_nans", True)

def safe_log(x):
    return jnp.log(jnp.maximum(x, 1e-6))


class MLP(nn.Module):
    features: Sequence[int]
    act: Callable = nn.gelu
    dtype : Any = jnp.float32

    @nn.compact
    def __call__(self, x, train=True):
        norm = partial(nn.BatchNorm,
                use_running_average=not train,
                momentum=0.9,
                epsilon=1e-4,
                dtype=self.dtype)


        for n, feat in enumerate(self.features[:-1]):
            x = nn.Dense(feat)(x)
            x = self.act(norm(name=f'norm_{n}')(x))

        x = nn.Dense(self.features[-1])(x)
        return x


class BernoulliSPG(nn.Module):
    dist: Callable
    mlp_hidden: Sequence[int] = (256,)*3
    min_pr: int = 0.1

    def setup(self, ):
        self.mlp = MLP(self.mlp_hidden+(2+self.dist.num_params,))

    @nn.compact
    def __call__(self, x, rng, train=False):
        return self.sample(x, rng, train=train)

    def _split_params(self, params):
        return params[0:2], params[2:]

    def sample(self, x, rng, train=False):
        dist_params = self.mlp(x, train=train)

        logit_params, dist_params = self._split_params(dist_params)
        p_d, _ = nn.softmax(logit_params)

        rng, rng_rd = random.split(rng, num=2)
        p_rain, p_dist = random.uniform(rng_rd, (2,), dtype=x.dtype)

        return jax.lax.cond(
            p_rain <= p_d,
            lambda: jnp.zeros(1, dtype=x.dtype)[0],
            lambda: self.dist.ppf(dist_params, p_dist)
        )

    def log_prob(self, x, y, train=True):
        dist_params = self.mlp(x, train=train)

        logit_params, dist_params = self._split_params(dist_params)
        p_d, p_r = nn.log_softmax(logit_params)

        return jax.lax.cond(
            y <= self.min_pr,
            lambda: p_d,
            lambda: p_r + self.dist.log_prob(dist_params, y)
        )


class Dist():

    def __init__(self, tfp_dist, num_params, param_func=None):
        self.num_params = num_params
        self.dist = tfp_dist
        if param_func is None:
            def param_func(x): return x
        self.param_func = param_func

    def __call__(self, params, prob):
        return self.ppf(self, params, prob)

    def log_prob(self, params, data):
        params = self.param_func(params)
        return self.dist(*params).log_prob(data)

    def prob(self, params, data, ):
        params = self.param_func(params)
        return self.dist(*params).prob(data)

    def ppf(self, params, prob, eps=1e-12):
        params = self.param_func(params)
        return self.dist(*params).quantile(prob)


Gamma = partial(Dist, num_params=2, param_func=lambda p: jax_utils.pos_only(p),  tfp_dist=tfd.Gamma)

Weibull = partial(Dist, num_params=2, param_func=lambda p: jax_utils.pos_only(p), tfp_dist=tfd.Weibull)

def gen_parto_func(params):
    return jnp.asarray([0.0, jax_utils.pos_only(params[0]), jax_utils.pos_only(params[1])]) 

GenPareto = partial(Dist, num_params=2, param_func=gen_parto_func, tfp_dist=tfd.GeneralizedPareto) 

# class BernoulliDist():
#     def sample(self, x, rng):
#         dist_params = self.mlp(x)

#         logit_params, dist_params = self._split_params(dist_params)
#         p_d, _ = nn.softmax(logit_params)

#         rng, rng_rd = random.split(rng, num=2)
#         p_rain, p_dist = random.uniform(rng_rd, (2,), dtype=x.dtype)

#         return jax.lax.cond(
#             p_rain <= p_d,
#             lambda: jnp.zeros(1, dtype=x.dtype)[0],
#             lambda: self.dist.ppf(dist_params, p_dist)
#         )

#     def log_prob(self, x, y):

#         dist_params = self.mlp(x)

#         logit_params, dist_params = self._split_params(dist_params)
#         p_d, p_r = nn.log_softmax(logit_params)

#         return jax.lax.cond(
#             y <= self.min_pr,
#             lambda: p_d,
#             lambda: p_r + self.dist.log_prob(dist_params, y)
#         )

class MixtureModel():
    def __init__(self, dists):
        self.dists = dists
        self.num_params: int = 0

        assert len(
            self.dists) > 0, 'Need at least one distribution for mixture model'

        # Find the total number of params in all the distributions
        for dist in self.dists:
            self.num_params += dist.num_params

        self.num_dists = len(self.dists)

        # One extra param for every dist for the weighting
        self.num_params += self.num_dists

    def _split_params(self, params):
        return params[0:self.num_dists], params[self.num_dists:]

    def _get_weightning_params(self, params):
        logit_params, dist_params = self._split_params(params)
        weightings = nn.softmax(logit_params*20)
        return weightings, dist_params

    def _loop_dist_func(self, params, dist_func: Callable):
        weightings, dist_params = self._get_weightning_params(params)

        output = 0.0
        last_idx = 0
        # Loop over each distribution and do a weighted sum
        for w, dist in zip(weightings, self.dists):
            next_idx = dist.num_params + last_idx
            p_dist = dist_params[last_idx:next_idx]
            last_idx = next_idx

            output += w*dist_func(dist, p_dist)

        # id_print(output)

        return output

    def ppf(self, params, prob):
        def dist_func(dist, p_dist):
            return dist.ppf(p_dist, prob)

        return self._loop_dist_func(params, dist_func)

    def log_prob(self, params, y):
        def dist_func(dist, p_dist):
            # In this case it is more simple to calculate the probability, then log later.
            return dist.prob(p_dist, y)

        prob = self._loop_dist_func(params, dist_func)
        return safe_log(prob)

def gamma_mix(num_dists=2):
    return MixtureModel(dists=[Gamma() for _ in range(num_dists)])
