import numpy as np
from .distributions import Dist
from jax import random
import jax.numpy as jnp


def cycle(arr, val):
    """ Removes the first value from arr and adds a new value to the end """
    arr = arr.at[:-1].set(arr[1:])
    return arr.at[-1].set(val)


class SPG():
    def __init__(self, rainday: Dist, rain_dists: dict, random_key: random.PRNGKey):
        self.rainday = rainday
        self.dists = rain_dists
        self.rnd_key = random_key

        self.dist_thresh = np.array(sorted(rain_dists.keys()))
        assert self.dist_thresh.min() == 0 and self.dist_thresh.max() < 1.0

    def _select_dist(self, prob, eps=1e-14):
        assert prob >= 0 and prob <= 1.0
        prob = jnp.clip(prob, 0+eps, 1-eps)

        idx = np.where(prob > self.dist_thresh)[0][-1]
        key = self.dist_thresh[idx]

        return self.dists[key]

    def sample(self, cond):
        prob_rain, prob_amount = random.uniform(self.rnd_key, (2,))

        is_rain = self.rainday.ppf(prob_rain, cond['rainday'])
        if is_rain:
            rain = self.select_dist(prob_amount, cond['rain'])
        else:
            rain = 0.0

        cond_next = {k: cycle(cond[k], v) for k, v in zip(['rainday', 'rain'], [is_rain, rain])}
        return rain, cond_next

    def generate(self, num_steps, cond_init: dict):
        data_out = []
        cond = cond_init

        for _ in range(num_steps):
            val, cond = self.sample(cond)
            data_out.append(val)

        return np.array(data_out)
