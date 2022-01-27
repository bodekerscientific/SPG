"""
    SPG using a single nn-module and a MLP for the distribution params
    
    @Author Leroy Bird
    
    export XLA_PYTHON_CLIENT_PREALLOCATE=false

"""

from typing import Sequence, Callable
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import elegy
import optax
from jax import random
import numpy as np


from spg import jax_utils, data_loader, data_utils
from bslibs.plot.qqplot import qqplot

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt


def safe_log(x):
    return jnp.log(jnp.maximum(x, 1e-6))


class MLP(nn.Module):
  features: Sequence[int]
  act : Callable = nn.gelu

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = self.act(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x


class BenoilSPG(nn.Module):
    dist : Callable
    mlp_hidden : Sequence[int] = (128,)
    min_pr : int = 0.1

    def setup(self, ):
        self.mlp = MLP(self.mlp_hidden+(4,))

    @nn.compact
    def __call__(self, x, rng):
        self.sample(x, rng)

    def _split_params(self, params):
        return params[:, 0:2], params[:, 2:]
    
    def sample(self, x, rng):
        dist_params = self.mlp(x)
        n = x.shape[0]

        logit_params, dist_params = self._split_params(dist_params)
        p_d, p_r = nn.softmax(logit_params).T

        rng, rng_rd = random.split(rng, num=2)
        p_rain, p_dist = random.uniform(rng_rd, (2, n), dtype=x.dtype)
        
        dd_mask = p_rain <= p_d

        output = jnp.zeros(n)
        output = output.at[dd_mask].set(0.0)
        output = output.at[~dd_mask].set(self.dist.ppf(dist_params[~dd_mask], p_dist[~dd_mask]))
        
        return output

    def log_prob(self, x, y):
        assert x.shape[0] == y.shape[0]

        dist_params = self.mlp(x)
        
        logit_params, dist_params = self._split_params(dist_params)
        p_d, p_r = nn.log_softmax(logit_params).T
        
        rd_mask = y <= self.min_pr

        output = jnp.zeros_like(y)
        output = output.at[rd_mask].set(p_d[rd_mask])
        output = output.at[~rd_mask].set(p_r[~rd_mask] + self.dist.log_prob(dist_params[~rd_mask], y[~rd_mask]))

        return output


class Gamma():
    def __init__(self,  num_params=2, param_init=None):
        self.num_params = 2
        self.param_func = jax_utils.pos_only
        self.dist = tfd.Gamma

    def log_prob(self, params, data, eps=1e-12):
        params = self.param_func(params) + eps
        return self.dist(*params.T).log_prob(data+eps)

    def ppf(self, params, probs, eps=1e-12):
        params = self.param_func(params)
        return self.dist(*params.T).quantile(probs)


class MixtureModel(elegy.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def call(self, x):

        x = elegy.nn.Linear(64, name="backbone")(x)
        x = jax.nn.relu(x)

        y: np.ndarray = jnp.stack(
            [elegy.nn.Linear(2, name="component")(x) for _ in range(self.k)], axis=1,
        )

        # equivalent to: y[..., 1] = 1.0 + jax.nn.elu(y[..., 1])
        y = jax.ops.index_update(y, jax.ops.index[..., 1], 1.0 + jax.nn.elu(y[..., 1]))

        logits = elegy.nn.Linear(self.k, name="gating")(x)
        probs = jax.nn.softmax(logits, axis=-1)

        return y, probs


class MixtureNLL(elegy.Loss):
    def call(self, y_true, y_pred):
        y, probs = y_pred
        y_true = jnp.broadcast_to(y_true, (y_true.shape[0], y.shape[1]))

        return -safe_log(
            jnp.sum(
                probs
                * jax.scipy.stats.norm.pdf(y_true, loc=y[..., 0], scale=y[..., 1]),
                axis=1,
            ),
        )


def make_qq(preds : list, targets : list, epoch : int, output_folder = './results'):
    plt.figure(figsize=(12, 8))
    qqplot(np.array(targets), np.array(preds), num_highlighted_quantiles=10)
    plt.savefig(Path(output_folder) / f'qq_{epoch}.png')
    plt.close()


def train(model, tr_loader, valid_loader=None, bs=128, lr=1e-4, num_epochs=100,  min_pr=0.1):

    rng = random.PRNGKey(42)
    x = jnp.zeros((bs, 2), dtype=jnp.float32)

    params = model.init(random.PRNGKey(0), x, rng)
    opt = optax.radam(lr)
    opt_state = opt.init(params)
    
    def loss_func(params, x, y):
        # Negative Log Likelihood loss
        probs = model.apply(params, x, y, method=model.log_prob)
        return (-probs).mean()

    def step(x, y, params, opt_state):
        loss, grads = jax.value_and_grad(loss_func)(params, x, y)   
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    for epoch in range(1, num_epochs+1):
        tr_loss = []
        for n, batch in enumerate(tr_loader):
            loss, params, opt_state = step(batch[0], batch[1], params, opt_state)
            tr_loss.append(loss)
        
        # Calculate the validation loss and generate some samples.
        val_loss = []
        preds = []
        targets = []
        for batch in valid_loader:
            val_loss.append(loss_func(params, *batch))
            rng, sample_rng = random.split(rng)
            x, y = batch
            preds.append(model.apply(params, x, sample_rng, method=model.sample))
            targets.append(y)

        preds = jnp.concatenate(preds, axis=None)
        targets = jnp.concatenate(preds, axis=None)
        make_qq(preds, targets, epoch)
        
        # Calculate the number of rain days
        dd_target = (targets < min_pr).sum()/targets.size
        dd_pred = (preds < min_pr).sum()/preds.size

        print(f'{epoch}/{num_epochs}, valid loss : {jnp.stack(val_loss).mean()}, train loss' \
              f' {jnp.stack(tr_loss).mean()}, dry day {dd_pred}, expected {dd_target}' )


if __name__ == '__main__':
    model = BenoilSPG(Gamma(), min_pr=0.1)
    data = data_utils.load_data()

    bs = 256
    tr_loader, val_loader = data_loader.get_data_loaders(data, bs=bs)
    train(model, tr_loader, val_loader, bs=bs)

    # rng = random.PRNGKey(42)
    # y = jnp.array([1.0, 2.0, 0, 0.04, 1.0, 1.0]).astype(jnp.float32)
    # x = y[:, None]

    # variables = model.init(random.PRNGKey(0), x, rng)
    
    # probs = model.apply(variables, x, y, method=model.log_prob)