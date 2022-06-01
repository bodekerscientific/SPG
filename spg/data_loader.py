from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
import json
from functools import partial

from spg import data_utils
from bslibs.regression.datetime_utils import datetimes_to_dec_year

def dt_360_to_dec_year(dts):
    return np.array([dt.year + ((dt.month-1)*30 + dt.day)/360.0 for dt in dts])

def generate_features_daily(pr : pd.Series, average_days=[1, 2, 4, 8], inc_doy=True, inc_tprime=True, rd_thresh=1.0, sce='ssp245', is_wh=False):
    features = []
    # Select y, and remove the last day as there would be no label        
    
    y = pr[1:]

    if is_wh:
        tp = y['tp'].values
        dts = y['dts'].values
        y = y['pr'].values
        pr = pr['pr']
    else:
        dts = y.index
        pr = pr

    pr = pr[:-1]
    pr_re_rd = (pr > rd_thresh).astype(np.float32)

    # Decimal day of the year
    if inc_doy:
        if is_wh:
            doy_frac = dt_360_to_dec_year(dts) % 1.0 
        else:
            # Mod 1 as we only want the fraction
            doy_frac = datetimes_to_dec_year(dts) % 1.0 
            
        # Use cos and sin doy to remove the jump at the start of a year
        features.extend([np.cos(2*np.pi*doy_frac), np.sin(2*np.pi*doy_frac)])
    
    # Calculate the n-day average across x and y
    for av_hr in average_days:
        features.append(pr.rolling(av_hr).mean().values) 
        features.append(pr_re_rd.rolling(av_hr).mean().values)
    
    if inc_tprime:
        if is_wh:
            features.append(tp)
        else:
            df_magic = data_utils.load_magic()
            features.append(data_utils.get_tprime_for_times(dts, df_magic[sce]))

    x = np.stack(features, axis=1).astype(np.float32)

    # Remove all the nans        
    mask = ~(np.isnan(x).any(axis=1) | np.isnan(y))

    return x[mask, :], y[mask]
 

def generate_features( pr : pd.Series, average_hours=[1, 3, 8, 24, 24*2, 24*6], inc_doy=True, inc_tod=True, inc_tprime=True, rd_thresh=0.1, sce='ssp245'):
    features = []
    # Select y, and remove the last day as there would be no label        
    y = pr[1:]
    dts = y.index
    y = y.values.astype(np.float32)

    pr = pr[:-1]
    pr_re_rd = (pr > rd_thresh).astype(np.float32)

    # Decimal day of the year
    if inc_doy:
        # Mod 1 as we only want the fraction
        doy_frac = datetimes_to_dec_year(dts) % 1.0 

        # Use cos and sin doy to remove the jump at the start of a year
        features.extend([np.cos(2*np.pi*doy_frac), np.sin(2*np.pi*doy_frac)])
    
    # Time of the day
    if inc_tod:
        hour_frac = dts.hour/24.0
        features.extend([np.cos(2*np.pi*hour_frac), np.sin(2*np.pi*hour_frac)])

    for av_hr in average_hours:
        features.append(pr.rolling(av_hr).mean().values) 
        features.append(pr_re_rd.rolling(av_hr).mean().values)
    
    if inc_tprime:
        df_magic = data_utils.load_magic()
        features.append(data_utils.get_tprime_for_times(dts, df_magic[sce]))

    x = np.stack(np.stack(features), axis=1).astype(np.float32)

    # Remove all the nans        
    mask = ~(np.isnan(x).any(axis=1) | np.isnan(y))

    return x[mask, :], y[mask]

def generate_features_multiscale(pr, max_hrs=24, pr_freq='H', cond_hr=4):
    output = defaultdict(list)
    # Fill missing times with nan
    pr = pr.resample(pr_freq).asfreq()
    #pr_rd = (pr > rd_thresh).astype(np.float32)

    for n in range(2, max_hrs+1):
        pr_av = pr.rolling(n).sum().values[n-1:]
        pr_av[pr_av < 1e-4] = 0
        
        pr_sub =  pr[:-n+1].values
        pr_sub[pr_sub < 1e-4] = 0
        
        x = []
        for cond_n in range(cond_hr):
            x.append(pr_sub[cond_n: -cond_hr + cond_n])
        x = np.stack(x, axis=1)
        
        pr_sub = pr_sub[cond_hr:]
        pr_av = pr_av[cond_hr:]
        
        assert len(pr_sub) == len(pr_av)
        assert len(pr_sub) == x.shape[0]
        
        mask = ~np.isnan(pr_av) & (pr_av > 0) & ~np.isnan(x).any(axis=1)

        pr_sub = pr_sub[mask]
        pr_av = pr_av[mask]
        x = x[mask, :]
        
        ratio = 1.0 - pr_sub / pr_av
        assert ratio.max() <= 1.0 + 1e-12
        assert ratio.min() + 1e-12 >= 0
        
        # Some values may be slightly above one due to rounding errors 
        ratio[ratio > 1.0 - 1e-6] = 1.0
        ratio[ratio < 1e-6] = 0.0

        output['freq'].extend(np.full_like(ratio, n))
        output['x'].extend(x)
        output['pr'].extend(pr_av)
        output['ratio'].extend(ratio)

    output = {k : np.stack(v) for k,v in output.items()}

    assert output['ratio'].min() == 0.0
    assert output['ratio'].max() == 1.0
    

    return output

def generate_features_split(pr, sum_period=24, pr_freq='H', cond_hr=12, eps=1e-5):
    pr = pr.resample(pr_freq).asfreq()
    pr_av = pr.rolling(sum_period).sum().values[sum_period - 1:]
    
    # y is the ratio of precip over sum_period, should alway sum to one.
    y = []
    for n in range(sum_period):
        end = -sum_period + n + 1
        if end < 0:
            pr_sub = pr[n:end].values
        else:
            pr_sub = pr[n:].values
            
        ratio = pr_sub/pr_av
        y.append(ratio)
    y = np.stack(y, axis=1)
    
    # Add the current amount of sum_period precipitation, next day, and fist day
    x = [pr_av[:-2*sum_period], pr_av[sum_period:-sum_period], pr_av[2*sum_period:]]
    x = np.stack(x, axis=1)
    
    assert cond_hr <= sum_period
    # Add the last n hours of precipitation
    x_cond = []
    for cond_n in range(cond_hr):
        x_cond.append(pr[sum_period - cond_hr + cond_n: -2*sum_period + cond_n + 1 - cond_hr].values)
    x = np.concatenate([x, np.stack(x_cond, axis=1)], axis=1)
    
    y = y[sum_period: -sum_period]
    pr_av = pr_av[sum_period: -sum_period]
    
    mask = ~np.isnan(pr_av) & (pr_av > 1e-4) & ~np.isnan(x).any(axis=1) #& ~np.isnan(y).any(axis=1)
    y = y[mask, :]
    x = x[mask, :]
    pr_av = pr_av[mask]
    
    assert ~np.isnan(y).any(axis=None)
    y[y > 1] = 1.0
    y[y < 0] = 0.0
    
    y = y + eps
    y = y / y.sum(axis=1)[:, None]
    
    return {'x' : x, 'ratio' : y, 'pr' : pr_av}

def calculate_stats(x, y):
    x_stats = {'mean' : np.mean(x, axis=0), 'std' : np.std(x, axis=0)}
    y_stats = {'mean' : np.array(0.0, dtype=np.float32), 'std' : np.std(y, axis=0)}
    return {'x' : x_stats, 'y' : y_stats}


def open_stats(stats_path):
    print(f'Loading stats from {stats_path}')
    with open(stats_path, 'r') as f:
        stats = json.loads(f.read())

        # Convert back into a dictionary
        func_np = partial(np.array, dtype=np.float32)
        return  {k: {'mean': func_np(v['mean']),
                    'std': func_np(v['std'])} for k, v in stats.items()}

def save_stats(stats, stats_path):
    print(f'Saving stats to {stats_path}')
    with open(stats_path, 'w') as f:
        # Can't save np array to json so convert to a list first
        f.write(json.dumps({k: {'mean': v['mean'].tolist(),
                                'std':  v['std'].tolist()} for k, v in stats.items()}, indent=4))
    return stats


def apply_stats(stats, data):
    return (data - stats['mean'])/stats['std']

def inverse_stats(stats, data):
    return data*stats['std'] + stats['mean']

class PrecipitationDataset(Dataset):
    def __init__(self, pr : pd.Series, stats=None, freq='H', **kwargs):
        #assert freq in ['H', 'D'], 'Only hourly or daily data supported'

        pr_re = pr.resample(freq, origin='start').asfreq()
        print(f' {(pr_re.isna().sum() / pr_re.size)*100:.2f}% of the values are missing.')
        
        self.X, self.Y = generate_features(pr_re, **kwargs) if freq == 'H' else generate_features_daily(pr_re, **kwargs)
        print(f'{(len(self.Y)/len(pr_re))*100:.2f}% of the values are valid after taking calculating the averages')

        if stats is None:
            print('Calculating stats')
            stats = calculate_stats(self.X, self.Y)

        self.stats = stats

    def inverse_tr(self, y, key='y'):
        return inverse_stats(self.stats[key], y)
        
    def apply_tr(self, data, key='x'):
        return apply_stats(self.stats[key], data)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.apply_tr(self.X[index]), self.apply_tr(self.Y[index], key='y') 


class PrecipitationDatasetWH(PrecipitationDataset):
    def __init__(self, pr : list, stats=None, **kwargs):
        x_lst = []
        y_lst = []
        for pr_df in pr:
            x, y = generate_features_daily(pr_df, is_wh=True, **kwargs)
            x_lst.append(x)
            y_lst.append(y)

        self.X = np.concatenate(x_lst, axis=0)
        self.Y = np.concatenate(y_lst, axis=0)

        if stats is None:
            print('Calculating stats')
            stats = calculate_stats(self.X, self.Y)
        self.stats = stats


class PrecipitationDatasetMultiScale(PrecipitationDataset):
    def __init__(self, pr, stats=None, stats_sample=5000, **kwargs):
        self.data = generate_features_split(pr)
        self.stats = stats
        
        if self.stats is None:
            assert stats_sample <= len(self)
            idxs = np.random.choice(len(self), size=stats_sample)

            x = np.stack([self[n][0] for n in idxs], axis=0)
            self.stats = {'x' : {'mean' : np.mean(x, axis=0), 'std' : np.std(x, axis=0)}}

    def apply_tr(self, data):
        if self.stats is None:
            return data
        else:
            return apply_stats(self.stats['x'], data)
 
    def __len__(self):
        return len(self.data['ratio'])

    def __getitem__(self, index):
        data = {k : v[index] for k,v in self.data.items()} 
        #x = np.concatenate([data['x'], [data['freq']], [data['pr']]])
        return self.apply_tr(data['x']), data['ratio'] 

@jax.jit
def jax_batch(batch):
    x = jnp.stack([b[0] for b in batch]).astype(jnp.float32)
    y = jnp.stack([b[1] for b in batch]).astype(jnp.float32)
    return x,y

def get_scale(data, num_valid=10000):
    return data[0:-num_valid].values.std()

def get_datasets(data, num_valid=10000, is_wh=False, stats_path = None, 
                 load_stats=False, ds_cls=PrecipitationDataset, **kwargs):
    if is_wh:
        # for weather@home, we split the dataset by ens for each batch (3 runs in total)
        assert len(data) % 3 == 0
        step_n = len(data) // 3

        # Split the data into 3 separate groups for each batch
        ens_split = [data[n*step_n:(n+1)*step_n] for n in range(3)]
        data_train = []
        data_valid = []
        for ens in ens_split:
            data_train.extend(ens[0:-num_valid])
            data_valid.extend(ens[-num_valid:])
    else:
        data_train = data[0:-num_valid]
        data_valid = data[-num_valid:]

    if load_stats:
        stats = open_stats(stats_path)
    else:
        stats = None

    train_ds = ds_cls(data_train, stats=stats, **kwargs)
    valid_ds = ds_cls(data_valid, stats=train_ds.stats, **kwargs)
    
    print(f'{len(train_ds)} items in the training dataset and {len(valid_ds)} items in the validation dataset')

    if not load_stats and stats_path is not None:
        save_stats(train_ds.stats, stats_path)

    return train_ds, valid_ds

def get_data_loaders(train_ds, valid_ds, bs=256, collate_fn=jax_batch, **kwargs):
    train_dataloader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, **kwargs)
    valid_dataloader = DataLoader(valid_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, **kwargs)

    return train_dataloader, valid_dataloader