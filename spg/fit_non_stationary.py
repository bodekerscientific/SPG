from typing import Callable

import jax.numpy as jnp
import jax
from jax.scipy.optimize import minimize
from functools import partial
from jax import random
import flax.linen as nn
from torch.utils.data import Dataset

import jax.nn.initializers as init
import matplotlib.pyplot as plt

from spg import spg_dist, jax_utils, run, data_utils, data_loader
from spg.train_spg import train


jax.config.update('jax_platform_name', 'cpu')
#jax.config.update("jax_enable_x64", True)

def post_process(params, cond):
    # a, b = jnp.split(params, 2, axis=0)
    # params = a + b*cond
    return params#jax_utils.pos_only(params)


class SimpleParams(nn.Module):
  num_params: int
  
  @nn.compact
  def __call__(self, x):
    offset = self.param('offset', init.normal(stddev=1), (self.num_params,))
    trend = self.param('trend', init.normal(stddev=1), (self.num_params,))
    return offset + trend*x

class SimpleDataset(Dataset):
    def __init__(self, data, stats=None):
        self.y = data.values
        magic_df = data_utils.load_magic()
        self.x = data_utils.get_tprime_for_times(data.index, magic_df['ssp245'])    
        self.stats = stats # Not used!

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class DistModel(nn.Module):
    dist: Callable
    
    def setup(self, ):
        self.param_cls = SimpleParams(self.dist.num_params)

    @nn.compact
    def __call__(self, x, rng=None, train=False):
        return self.param_cls(x,)

    def sample(self, x, rng, train=False):
        dist_prams = self(x, rng)
        p_dist = random.uniform(rng, (1,))
        return self.dist.ppf(dist_prams, p_dist)
        
    def log_prob(self, x, y, train=False):
        dist_prams = self(x)
        return self.dist.log_prob(dist_prams, y + 1e-6)
    
    def cdf(self, x, y, train=False):
        dist_prams = self(x)
        return self.dist.cdf(dist_prams, y)

    def ppf(self, x, p, train=False):
        dist_prams = self(x)
        return self.dist.ppf(dist_prams, p)

    
def fit_ns(data, dist):
    tr_ds, val_ds = data_loader.get_datasets(data, num_valid=3000, ds_cls=SimpleDataset)
    train_dl, valid_dl = data_loader.get_data_loaders(tr_ds, val_ds)
    
    model = DistModel(dist)
    
    params = train(model, num_feat=1, tr_loader=train_dl, valid_loader=valid_dl, num_epochs=100,
                   opt_kwargs=dict(max_lr = 1e-2, spin_up_steps=50, max_steps=50*100, min_lr=1e-7, wd=1e-4))
    
    # if params_init is None:
    #     params_init = random.uniform(rng, (dist.num_params,))*0.4 + 0.5*jnp.ones(dist.num_params)
    
    # def loss_func(params, cond, data):
    #     wd = 0.5 * weight_decay * params@params
    #     params = post_process(params, cond)
    #     return -dist.log_prob(params, data + 1e-12) + wd 
    
    # @jax.jit
    # def _fit(params):
    #     map_func = partial(loss_func, params=params)
    #     return jax.vmap(map_func)(cond=cond_batch, data=data).mean()

    # res = minimize(_fit, params_init, method="BFGS")
    #assert res.success
    return params, model

    
def benchmark_mixtures(data, num_mix=5, thresh=.5):
    mask = data >= thresh
    data = data[mask] - thresh
    scale = data.values.std()
    data = data/scale

    #for dist in [distributions.TFGammaMixCond]:
    rng = jax.random.PRNGKey(42)
    dist = spg_dist.MixtureModel([spg_dist.Gamma(), spg_dist.GenPareto()])
    params, model = fit_ns(data, dist)
    
    def apply_func(*args,  method=model.sample):
        return model.apply(params, *args,  method=method)

    def apply_func_vmap(*args, rng=None, method=model.sample):
        func = partial(model.apply, params, method=method)
        return jax.vmap(func)(*args)
    
    magic_df = data_utils.load_magic()
    t_prime = data_utils.get_tprime_for_times(data.index, magic_df['ssp245'])

    probs = random.uniform(rng, shape=t_prime.shape)
    sample = apply_func_vmap(t_prime, probs, method=model.ppf)
    run.plot_qq(data, sample, output_path=f'qq_mix.png')
    
    
    scale_tprime = t_prime.max() -  t_prime.min()
    
    quantiles = jnp.linspace(0, 1.0, 10000, endpoint=False)
    sample_zero = apply_func_vmap(jnp.full_like(quantiles, t_prime.min()), quantiles, method=model.ppf) + thresh
    sample_one = apply_func_vmap(jnp.full_like(quantiles, t_prime.max()), quantiles, method=model.ppf) + thresh
    change = (100/scale_tprime)*(sample_one - sample_zero)/sample_zero
    print(change[-1])
    plt.figure(figsize=(12,8))
    plt.plot(quantiles, change)
    plt.ylabel('Percentage change')
    plt.savefig('curve.png')
    plt.close()
    
    plt.figure(figsize=(12,8))
    plt.title('Percentage change per degree')
    sample_zero = apply_func_vmap(jnp.zeros_like(quantiles), quantiles, method=model.ppf) + thresh
    for t in jnp.arange(0.2, 1.7, 0.2):
        quantiles = jnp.linspace(0, 1.0, 10000, endpoint=False)
        sample_one = apply_func_vmap(jnp.full_like(quantiles, t), quantiles, method=model.ppf) + thresh
        
        change = (100/t)*(sample_one - sample_zero)/sample_zero
        plt.plot(quantiles, change, label=f'Tprime = 0K to {t}')
    plt.legend()
    plt.savefig('curves.png')

if __name__ == '__main__':
    data = data_utils.load_data()
    benchmark_mixtures(data)