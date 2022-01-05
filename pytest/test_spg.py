from pathlib import Path
import jax.numpy as jnp
from jax import random

from spg.generator import SPG, cycle
from spg.distributions import Dist, RainDay
from spg import distributions
from spg.run import load_data, plot_qq
import matplotlib.pyplot as plt
import scipy.stats as ss

_SEED = 42


def _get_rnd_key():
    return random.PRNGKey(_SEED)

def test_cylc():
    a = jnp.array([1, 5 ,2, 1])
    a = cycle(a, 10)
    a = cycle(a, 1)
    assert (a == jnp.array([2, 1, 10, 1])).all()

def test_select_dist():
    a,b,c,d = Dist('a', None), Dist('b', None), Dist('c', None), Dist('d', None)
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

def _get_preds(sp, num_steps=1000):
    ar_depth = sp.rainday.ar_depth
    cond = jnp.ones(ar_depth)
    cond = {'rain': None, 'rainday': jnp.ones([1, ar_depth])}
    #time = pd.date_range(start=data.index[0], end='2050-1-1')
    return sp.generate(num_steps=num_steps, cond_init=cond)

def test_simple_spg(data = load_data()):
    #data[data > 50] = 0

    thresh = 0.2
    rd = distributions.RainDay(thresh=thresh, ar_depth=1)

    rng = random.PRNGKey(42)
    sp_tf = SPG(rd, {0: distributions.TFWeibull()}, rng)
    sp_tf.fit(data.values)

    rng = random.PRNGKey(42)
    sp_ss = SPG(rd, {0: distributions.SSWeibull()}, rng)
    sp_ss.fit(data.values)

    sp_ss.print_params()
    sp_tf.print_params()

    preds_ss = _get_preds(sp_ss)
    preds_tf = _get_preds(sp_tf)
    
    ss_dist = ss.weibull_min
    data_sub = data[data >= thresh] - thresh + 1e-12

    print(len(data_sub))
    print(ss_dist.fit(data_sub, floc=0))

    # print(preds_ss)
    # print(preds_tf)

    # print(preds_tf - preds_ss)
    
    plt.figure(figsize=(20,14))
    plt.scatter(preds_tf, preds_ss)
    plt.savefig('scatter.png', dpi=300)

    plot_qq(data, preds_ss, Path('./qq_ss.png'))
    plot_qq(data, preds_tf, Path('./qq_tf.png'))

if __name__ == '__main__':
    test_simple_spg()