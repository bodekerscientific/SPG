import numpy as np
from spg.distributions import Dist 

from jax import random
import jax.numpy as jnp


def cycle(arr, val):
    """ Removes the first value from arr and adds a new value to the end """
    val = jnp.atleast_1d(val)
    if arr.ndim != val.ndim:
        val = jnp.atleast_2d(val)
        assert val.ndim == val.ndim
        
    arr = arr.at[..., :-1].set(arr[..., 1:])
    return arr.at[..., -1:].set(val)


class SPG():
    def __init__(self, rainday: Dist, rain_dists: dict, random_key: random.PRNGKey):
        self.rainday = rainday
        self.dists = rain_dists
        self.rnd_key = random_key
        self.thresholds = None
        self.offset_dist = {}

        self.dist_thresh = np.array(sorted(rain_dists.keys()))
        assert self.dist_thresh.min() == 0 and self.dist_thresh.max() < 1.0

    def _select_dist(self, prob, eps=1e-14):
        assert prob >= 0 and prob <= 1.0
        prob = jnp.clip(prob, 0+eps, 1-eps)

        idx = np.where(prob > self.dist_thresh)[0][-1]
        key = self.dist_thresh[idx]

        return self.dists[key]

    def sample(self, cond):
        self.rnd_key, subkey = random.split(self.rnd_key)
        prob_rain, prob_sel, prob_dist = random.uniform(subkey, (3,))
    
        is_rain = self.rainday.ppf(prob_rain, cond['rainday'])
        if is_rain:
            dist = self._select_dist(prob_sel)
            rain = dist.ppf(prob_dist, cond['rain']) + dist.offset + self.rainday.thresh
        else:
            rain = 0.0

        cond_next = {k: cycle(cond[k], v) if cond[k] is not None else None 
                    for k, v in zip(['rainday', 'rain'], [is_rain, rain])}

        return rain, cond_next

    def generate(self, num_steps, cond_init: dict):
        data_out = []
        cond = cond_init

        for _ in range(num_steps):
            val, cond = self.sample(cond)
            data_out.append(val)

        return np.array(data_out)

    def fit(self, data):
        self.rainday.fit(data)
        
        data = data[data >= self.rainday.thresh] - self.rainday.thresh
        self.thresholds = np.quantile(data, list(self.dist_thresh) + [1.0])
        
        # Ensure we don't miss any data
        thresh = self.thresholds.copy()
        thresh[0] -= 1
        thresh[-1] = np.inf

        for lower, upper, key in zip(self.thresholds[:-1], self.thresholds[1:], self.dist_thresh):
            data_sub = data[(data>=lower)]
            print(f'Fitting dist, q={key}, from {lower} to {upper} with {len(data_sub)} datapoints')
            self.dists[key].fit(data_sub - lower)
            self.dists[key].offset = lower
            
    def print_params(self, ):
        print(self.rainday)
        for d in self.dists.values():
            print(d)
        



