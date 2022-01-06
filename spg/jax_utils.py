import scipy.stats as ss
import numpy as np
from jax.scipy.optimize import minimize
import jax.numpy as jnp
import jax
from jax import nn


def lin_regress(x, y):
    return jnp.linalg.inv(x.T @ x) @ (x.T @ y)

def logistic_regress(x, w):
    return nn.sigmoid(x @ w)

def prob_bin(probs, thresh=0.5):
    """ Calculates binary predictions """
    pred_bin = jnp.array(probs)
    pred_bin = jnp.where(pred_bin > thresh, pred_bin, 0.0)
    pred_bin = jnp.where(pred_bin <= thresh, pred_bin, 1.0)
    return pred_bin

def linear_expansion(params, cond, key='tprime'):
    return params['offset'][..., None] + params['coefs'][..., None]*cond[key][None]

def pos_only(x):
    """ Ensures the values of x always positive"""
    return nn.elu(x) + 1.0

def apply_pos(arr, idx):
    return arr.at[idx].set(pos_only(arr[idx]))

def linear_exp_split(params, *args, func=linear_expansion, **kwargs):
    assert len(params) % 2 == 0
    p = jnp.split(params, 2)
    params = {
        'offset' : p[0],
        'coefs' : p[1]
    }
    return func(params, *args, **kwargs)

def logistic_loss(w, x, y, eps=1e-14, weight_decay=0.1):
    """ Generic nll with regularization from 
        https://www.architecture-performance.fr/ap_blog/logistic-regression-with-jax/"""
    #n = y.size
    p = logistic_regress(x, w)
    p = jnp.clip(p, eps, 1 - eps)  # bound the probabilities
    return -jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p)) + 0.5 * weight_decay * w@w


def nll_loss(preds, targets, eps=1e-14):
    p = jnp.clip(preds, eps, 1 - eps)
    y = jnp.clip(targets, eps, 1 - eps)
    return -jnp.sum(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))

