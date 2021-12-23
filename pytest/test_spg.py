from spg.spg import SPG, cycle
from spg.distributions import Dist, RainDay

import jax.numpy as jnp
from jax import random

_SEED = 42

def _get_rnd_key():
    return random.PRNGKey(_SEED)

def test_cylc():
    a = jnp.array([1, 5 ,2, 1])
    a = cycle(a, 10)
    a = cycle(a, 1)
    assert (a == jnp.array([2, 1, 10, 1])).all()

def test_select_dist():
    a,b,c,d = Dist(), Dist(), Dist(), Dist()
    rd = RainDay(ar_depth=0)
    dists = { 0.0 : a,
              0.5 : b,
              0.2 : c,
              0.99 : d,
    }

    pgen = SPG(rd,dists, _get_rnd_key)
    assert pgen._select_dist(0.) == a
    assert pgen._select_dist(0.15) == a
    assert pgen._select_dist(0.51) == b
    assert pgen._select_dist(0.3) == c
    assert pgen._select_dist(0.999) == d
    

def test_b_norm():
    pass

if __name__ == '__main__':
    test_select_dist()