import numpy as np
from spg.distributions import Dist 

from jax import random, jit
import jax.numpy as jnp


def cycle(arr, val):
    """ Removes the first value from arr and adds a new value to the end """
    val = jnp.atleast_1d(val)
    if arr.ndim != val.ndim:
        val = jnp.atleast_2d(val)
        assert val.ndim == val.ndim
        
    arr = arr.at[..., :-1].set(arr[..., 1:])
    return arr.at[..., -1:].set(val)


def apply_mask_to_dict(target, mask):
    if target is None:
        return None
    else:
        return {k : v[mask] if v.size == mask.size else v for k,v in target.items()}

def apply_func_to_dict(target, func):
    if target is None:
        return None
    else:
        return {k : func(v) for k,v in target.items()}

class SPG():
    def __init__(self, rainday: Dist, rain_dists: dict, random_key: random.PRNGKey, max_val=1000):
        self.rainday = rainday
        self.dists = rain_dists
        self.rnd_key = random_key
        self.thresholds = None
        self.offset_dist = {}
        self.max_val = max_val


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
            # Calculate the value using the inverse of the cdf (ppf), we scale the prob by the max value allowed
            # For the distribution.
            rain = dist.ppf(prob_dist*dist.max_prob, cond['rain']) + dist.offset
            assert jnp.isfinite(rain)
            # Transfor the standardized rain back
            rain = rain*self._scale + self.rainday.thresh
        else:
            rain = 0.0

        return rain

    def generate(self, num_steps, cond_init: dict, cond_func=None):
        if cond_func is None:
            cond_func = lambda x, y: y

        data_out = []
        cond = cond_init
        
        for _ in range(num_steps):
            val = self.sample(cond)
            data_out.append(float(val))
            cond = cond_func(data_out, cond)

        return np.stack(data_out)

    def fit(self, data, cond=None, use_max_prob=False):
        self.rainday.fit(data)
        
        # Subset the rain days only
        mask = data >= self.rainday.thresh
        cond = apply_mask_to_dict(cond, mask)

        data = data[mask] - self.rainday.thresh
        self._scale = data.std()
        data = data/self._scale

        self.thresholds = np.quantile(data, list(self.dist_thresh) + [1.0])
        
        # Ensure we don't miss any data
        thresh = self.thresholds.copy()
        thresh[-1] = self.max_val/self._scale

        for lower, upper, key in zip(self.thresholds[:-1], self.thresholds[1:], self.dist_thresh):
            mask_sub = (data>=lower) & (data<upper)

            data_sub = data[mask_sub]
            cond_sub = apply_mask_to_dict(cond, mask_sub)

            print(f'Fitting dist, q={key}, from {lower*self._scale} to {upper*self._scale} with {len(data_sub)} datapoints')
            
            self.dists[key].fit(data_sub - lower, cond_sub)
            # We need to remember the offset, to shift the data back
            self.dists[key].offset = lower

            # We save the max_prob, so we can generate values greater then the max val.
            if np.isfinite(upper) and use_max_prob:
                # Take the mean of the function for now.
                max_prob = self.dists[key].cdf(upper - lower, apply_func_to_dict(cond, jnp.mean))
                self.dists[key].max_prob = max_prob
                
    def print_params(self, ):
        print(self.rainday)
        for d in self.dists.values():
            print(d)
        



