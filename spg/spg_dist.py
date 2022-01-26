from typing import Sequence, Callable

import flax.linen as nn

import jax
import jax.numpy as jnp
import elegy
import optax
from jax import random

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
    mlp : Callable
    dist : Callable
    min_pr : 0.1

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
        p_rain, p_dist = random.uniform(rng_rd, (2, n))
        
        rd_mask = p_rain <= p_d

        output = jnp.zeros(n)
        output = output.at[rd_mask].set(0.0)
        output = output.at[~rd_mask].set(self.dist.ppf(dist_params, x, p_dist))
        
        return output

    def log_prob(self, x, y):
        assert x.shape[0] == y.shape[0]

        dist_params = self.mlp(x)
        
        logit_params, dist_params = self._split_params(dist_params)
        p_d, p_r = nn.log_softmax(logit_params).T
        
        rd_mask = y <= self.min_pr

        output = jnp.zeros_like(y)
        output = output.at[rd_mask].set(p_d)
        output = output.at[~rd_mask].set(p_r + self.dist.log_prob(dist_params[~rd_mask], x[~rd_mask], y[~rd_mask]))

        return output

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

if __name__ == '__main__':
    model = elegy.Model(
        module=MixtureModel(k=3), loss=MixtureNLL(), optimizer=optax.adam(3e-4)
    )

    model.summary(X_train[:batch_size], depth=1)

    model.fit(
        x=X_train, y=y_train, epochs=500, batch_size=batch_size, shuffle=True,
    )
