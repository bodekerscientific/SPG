"""
    SPG using a single nn-module and a MLP for the distribution params
    
    @Author Leroy Bird
    
    You might need to set, as TF and JAX will both try grab the full gpu memory.
        export XLA_PYTHON_CLIENT_PREALLOCATE=false

"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
from typing import Iterable, Sequence, Callable
from pathlib import Path

import flax
import flax.linen as nn
from jax.experimental.host_callback import id_print
import jax
import jax.numpy as jnp
import optax
from jax import random
import numpy as np
from functools import partial


from spg import jax_utils, data_loader, data_utils
from bslibs.plot.qqplot import qqplot

import wandb

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

matplotlib.use('AGG')


# Global flag to set a specific platform, must be used at startup.
#jax.config.update('jax_platform_name', 'cpu')


def safe_log(x):
    return jnp.log(jnp.maximum(x, 1e-6))


class MLP(nn.Module):
    features: Sequence[int]
    act: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            norm = nn.BatchNorm(use_running_average=True)
            #norm = nn.LayerNorm()#BatchNorm(use_running_average=True)
            x = self.act(norm(x))

        x = nn.Dense(self.features[-1])(x)
        return x


class BernoulliSPG(nn.Module):
    dist: Callable
    mlp_hidden: Sequence[int] = (256,)*3
    min_pr: int = 0.1

    def setup(self, ):
        self.mlp = MLP(self.mlp_hidden+(2+self.dist.num_params,))

    @nn.compact
    def __call__(self, x, rng):
        self.sample(x, rng)

    def _split_params(self, params):
        return params[0:2], params[2:]

    def sample(self, x, rng):
        dist_params = self.mlp(x)

        logit_params, dist_params = self._split_params(dist_params)
        p_d, _ = nn.softmax(logit_params)

        rng, rng_rd = random.split(rng, num=2)
        p_rain, p_dist = random.uniform(rng_rd, (2,), dtype=x.dtype)

        return jax.lax.cond(
            p_rain <= p_d,
            lambda: jnp.zeros(1, dtype=x.dtype)[0],
            lambda: self.dist.ppf(dist_params, p_dist)
        )

    def log_prob(self, x, y):

        dist_params = self.mlp(x)

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

    def log_prob(self, params, data, eps=1e-12):
        params = self.param_func(params)
        return self.dist(*params).log_prob(data+eps)

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

    def log_prob(self, params, y, eps=1e-6):
        def dist_func(dist, p_dist):
            # In this case it is more simple to calculate the probability, then log later.
            return jnp.exp(dist.log_prob(p_dist, y))

        prob = self._loop_dist_func(params, dist_func)
        return jnp.log(prob + eps)


def make_qq(preds: list, targets: list, epoch: int, log, output_folder='./results/plots'):
    plt.figure(figsize=(12, 8))
    qqplot(np.array(targets), np.array(preds), num_highlighted_quantiles=10, linewidth=0)
    output_path = Path(output_folder) / f'qq_{str(epoch).zfill(3)}.png'
    plt.savefig(output_path, transparent=False, dpi=150)
    plt.close()

    log({'qq_plot' : wandb.Image(str(output_path))})

def save_params(params, epoch : int, output_folder='./results/params'):
    output_path = Path(output_folder) / f'params_{str(epoch).zfill(3)}.data'
    
    params_bytes = flax.serialization.to_bytes(params)
    with open(output_path, 'wb') as f:
        f.write(params_bytes)
    
    return output_path

def get_opt(params, max_lr):
    sched = optax.warmup_cosine_decay_schedule(1e-5, max_lr, 2000, 40000, 1e-5)
    opt = optax.adamw(sched, weight_decay=1e-4)
    opt = optax.apply_if_finite(opt, 20)
    params =  optax.LookaheadParams(params, deepcopy(params))
    opt = optax.lookahead(opt, 5, 0.5)
    opt_state = opt.init(params)

    return opt, opt_state, params

def train(model, num_feat, log, tr_loader, valid_loader, params=None, max_lr=1e-3, num_epochs=100, min_pr=0.1):
    rng = random.PRNGKey(42**2)
    x = jnp.zeros((num_feat,), dtype=jnp.float32)

    params_init = model.init(random.PRNGKey(0), x, rng)
    if params is None:
        params = params_init

    opt, opt_state, params = get_opt(params, max_lr)

    def loss_func(params, x_b, y_b):
        # Negative Log Likelihood loss
        def nll(x, y):
            return -model.apply(params, x, y, method=model.log_prob)

        # Use vamp to vectorize over the batch dim.
        return jax.vmap(nll)(x_b, y_b).mean()

    @jax.jit
    def step(x, y, params, opt_state):
        loss, grads = jax.value_and_grad(loss_func)(params.fast, x, y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    
    @jax.jit
    def val_step(x_b, y_b, params, rng):
        loss = loss_func(params.slow, x_b, y_b)

        def get_sample(x, rng):
            rng, sample_rng = random.split(rng)
            return model.apply(params.slow, x, sample_rng, method=model.sample)

        rngs = random.split(rng, num=len(x_b))
        pred = jax.vmap(get_sample)(x_b, rngs)
        return loss, pred, random.split(rng)[1]

    best_loss = jnp.inf
    for epoch in range(1, num_epochs+1):
        tr_loss = []
        for n, batch in enumerate(tr_loader):
            #assert not np.isnan(batch[0]).any()
            loss, params, opt_state = step(batch[0], batch[1], params, opt_state)

            #id_print(loss)
            wandb.log({'train_loss_step' : loss})
            tr_loss.append(loss)
        
        param_path = save_params(params.slow, epoch)

        # Calculate the validation loss and generate some samples.
        val_loss = []
        preds = []
        targets = []
        for batch in valid_loader:
            x, y = batch
            loss, pred, rng = val_step(x, y, params, rng)
            val_loss.append(loss)
            preds.append(pred)
            targets.append(y)

        preds = jnp.concatenate(preds, axis=None)
        targets = jnp.concatenate(targets, axis=None)

        # Calculate the expected number of rain days
        dd_target = (targets < min_pr).sum()/targets.size
        dd_pred = (preds < min_pr).sum()/preds.size
        
        val_loss = jnp.stack(val_loss).mean()
        tr_loss = jnp.stack(tr_loss).mean()

        log({'train_loss_epoch': tr_loss,
             'val_loss': val_loss,
             'epoch': epoch,
             'dd_val' : dd_pred,
             'dd_mae' : jnp.abs(dd_target - dd_pred)})

        print(f'{epoch}/{num_epochs}, valid loss : {val_loss:.4f}, train loss'
              f' {tr_loss:.4f}, dry day prob {dd_pred:.4f}, expected {dd_target:.4f}')

        make_qq(preds, targets, epoch, log)

        if epoch > 2 and val_loss < best_loss:
            best_loss = val_loss
            wandb.log_artifact( str(param_path), name='params_' + str(epoch).zfill(3), type='training_weights') 


def load_params(model, path, num_feat): 
    x = jnp.zeros((num_feat,), dtype=jnp.float32)
    params = model.init(random.PRNGKey(0), x, random.PRNGKey(42))

    print(f'Loading params from {path}...')
    with open(path, 'rb') as f:
        params = flax.serialization.from_bytes(params, f.read())

    return params


def gamma_mix(num_dists=2):
    return MixtureModel(dists=[Gamma() for _ in range(num_dists)])

# 
# def get_model():
#     return BernoulliSPG(dist=MixtureModel(dists=[Gamma(), GenPareto(), GenPareto()]))



def get_model():
    return BernoulliSPG(dist=MixtureModel(dists=[Weibull(), GenPareto(), GenPareto()]))


def train_wh(**kwargs):
    print('Loading weather@home')
    wh_ens = data_utils.load_wh(num_ens=500)
    print(f'Loaded {len(wh_ens)} ens')

    ds_train, ds_valid = data_loader.get_datasets(wh_ens, num_valid=50, is_wh=True)

    tr_loader, val_loader = data_loader.get_data_loaders(ds_train, ds_valid, bs=bs)
    num_feat = len(ds_train[0][0])
    print(f'Num features {num_feat}')

    train(num_feat=num_feat, tr_loader=tr_loader, valid_loader=val_loader, **kwargs)

def train_daily(model, load_stats=True, params_path=None, **kwargs):
    data = data_utils.load_data()
    
    ds_train, ds_valid = data_loader.get_datasets(data, num_valid=365*4, load_stats=load_stats, is_wh=False, freq='D')

    tr_loader, val_loader = data_loader.get_data_loaders(ds_train, ds_valid, bs=bs)
    num_feat = len(ds_train[0][0])
    print(f'Num features {num_feat}')

    if params_path is not None:
        params = load_params(model, params_path, num_feat)
    else:
        params = None

    train(model, num_feat=num_feat, tr_loader=tr_loader, valid_loader=val_loader, params=params, **kwargs)


if __name__ == '__main__':
    model = get_model()
    logging=True

    if logging:
        wandb.init(entity='bodekerscientific', project='SPG')
        logger = wandb.log
    else:
        logger = lambda *args : None

    bs = 256

    params_path = 'params_wh.data'

    train_daily(model=model, log=wandb.log, params_path=params_path)
    #train_wh(model=model, log
    # =wandb.log)
    
    # rng = random.PRNGKey(42)
    # y = jnp.array([1.0, 2.0, 0, 0.04, 1.0, 1.0]).astype(jnp.float32)
    # x = y[:, None]

    # variables = model.init(random.PRNGKey(0), x, rng)
    
    # probs = model.apply(variables, x, y, method=model.log_prob)
