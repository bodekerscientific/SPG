#%%
import jax.numpy as np
from jax import random

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from jax.experimental.host_callback import id_print
import numpy
#%%

from spg import data_utils
data = data_utils.load_data()
data = data.values
data = data[data > 0.1] - 0.1
scale = data.std()
data = data / scale
X = np.stack([data, np.ones_like(data)], axis=1)

#%%
# n_samples = 2000
# noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
# X, y = noisy_moons
# X = StandardScaler().fit_transform(X)
# xlim, ylim = [-2, 2], [-2, 2]
# plt.scatter(X[:, 0], X[:, 1], s=10, color='red')
# plt.xlim(xlim)
# plt.ylim(ylim)
# %%
# Initialize parameters, not committing to a batch shape
rng = random.PRNGKey(0)

def sample_n01(N):
  D = 2
  return random.normal(rng, (N, D))

def log_prob_n01(x):
  return np.sum(-np.square(x)/2 - np.log(np.sqrt(2*np.pi)), axis=-1)
# %%

# %%
from jax.experimental import stax # neural network library
from jax.experimental.stax import Dense, Relu # neural network layers
#%%
def nvp_forward(net_params, shift_and_log_scale_fn, x, flip=False):
  d = x.shape[-1]//2
  x1, x2 = x[:, :d], x[:, d:]
  if flip:
    x2, x1 = x1, x2
  shift, log_scale = shift_and_log_scale_fn(net_params, x1)
  y2 = x2*np.exp(log_scale) + shift
  if flip:
    x1, y2 = y2, x1
  y = np.concatenate([x1, y2], axis=-1)
  return y

def nvp_inverse(net_params, shift_and_log_scale_fn, y, flip=False):
  d = y.shape[-1]//2
  y1, y2 = y[:, :d], y[:, d:]
  if flip:
    y1, y2 = y2, y1
  shift, log_scale = shift_and_log_scale_fn(net_params, y1)
  x2 = (y2-shift)*np.exp(-log_scale)
  if flip:
    y1, x2 = x2, y1
  x = np.concatenate([y1, x2], axis=-1)
  return x, log_scale

def init_nvp():
  D = 2
  net_init, net_apply = stax.serial(
    Dense(512), Relu, Dense(512), Relu, Dense(D))
  in_shape = (-1, D//2)
  out_shape, net_params = net_init(rng, in_shape)
  def shift_and_log_scale_fn(net_params, x1):
    s = net_apply(net_params, x1)
    return np.split(s, 2, axis=1)
  return net_params, shift_and_log_scale_fn

def sample_nvp(net_params, shift_log_scale_fn, base_sample_fn, N, flip=False):
  x = base_sample_fn(N)
  return nvp_forward(net_params, shift_log_scale_fn, x, flip=flip)

def log_prob_nvp(net_params, shift_log_scale_fn, base_log_prob_fn, y, flip=False):
  x, log_scale = nvp_inverse(net_params, shift_log_scale_fn, y, flip=flip)
  ildj = -np.sum(log_scale, axis=-1)
  return base_log_prob_fn(x) + ildj

def init_nvp_chain(n=2):
  flip = False
  ps, configs = [], []
  for i in range(n):
    p, f = init_nvp()
    ps.append(p), configs.append((f, flip))
    flip = not flip
  return ps, configs

def sample_nvp_chain(ps, configs, base_sample_fn, N):
  x = base_sample_fn(N)
  for p, config in zip(ps, configs):
    shift_log_scale_fn, flip = config
    x = nvp_forward(p, shift_log_scale_fn, x, flip=flip)
  return x

def make_log_prob_fn(p, log_prob_fn, config):
  shift_log_scale_fn, flip = config
  return lambda x: log_prob_nvp(p, shift_log_scale_fn, log_prob_fn, x, flip=flip)

def log_prob_nvp_chain(ps, configs, base_log_prob_fn, y):
  log_prob_fn = base_log_prob_fn
  for p, config in zip(ps, configs):
    log_prob_fn = make_log_prob_fn(p, log_prob_fn, config)
  return log_prob_fn(y)

#%%
ps, cs = init_nvp_chain(4)
y = sample_nvp_chain(ps, cs, sample_n01, 1000)
a = log_prob_nvp_chain(ps, cs, log_prob_n01, X)

# %%
a


# %%
import numpy

plt.hist(numpy.array(a))
# %%

from jax.experimental import optimizers
from jax import jit, value_and_grad
import numpy as onp

ps, cs = init_nvp_chain(8)

@jit
def loss(params, batch):
  l = -np.mean(log_prob_nvp_chain(params, cs, log_prob_n01, batch))
  return l

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-4,)

def step(i, opt_state, batch):
  params = get_params(opt_state)
  l, g = value_and_grad(loss)(params, batch)
  id_print(l)
  if np.isfinite(l).all():
    return opt_update(i, g, opt_state)
  else:
    return opt_state
opt_state = opt_init(ps)

#%%
iters = int(8e4)
data_generator = (X[onp.random.choice(X.shape[0], 100)] for _ in range(iters))
for i in range(iters):
  opt_state = step(i, opt_state, next(data_generator))
ps = get_params(opt_state)
# %%
y = sample_nvp_chain(ps, cs, sample_n01, 1000)
y = numpy.array(y)

plt.scatter(y[:, 0], y[:, 1], s=10, color='red')
xlim, ylim = [-3, 3], [-3, 3]
# %%
from spg import data_utils
data = data_utils.load_data()
data = data.values
data = data[data > 0.1] - 0.1
scale = data.std()
data = data / scale
# %%
data = np.stack([data, np.ones_like(data)], axis=1)
# %%
data
# %%
data.max()
# %%
data.min()
# %%
