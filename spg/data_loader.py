from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax

from spg import data_utils
from bslibs.regression.datetime_utils import datetimes_to_dec_year


def dt_360_to_dec_year(dts):
    return np.array([dt.year + ((dt.month-1)*30 + dt.day)/360.0 for dt in dts])

def generate_features_daily(pr : pd.Series, average_days=[1, 2, 4, 8], inc_doy=True, inc_tprime=True, rd_thresh=0.1, sce='ssp245', is_wh=False):
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
    y = y.values

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

def calculate_stats(x, y):
    x_stats = {'mean' : np.mean(x, axis=0), 'std' : np.std(x, axis=0)}
    y_stats = {'mean' : 0, 'std' : np.std(y, axis=0)}
    return {'x' : x_stats, 'y' : y_stats}


class PrecipitationDataset(Dataset):
    def __init__(self, pr : pd.Series, stats=None, freq='H'):
        pr_re = pr.resample(freq).asfreq()
        print(f' {(pr_re.isna().sum() / pr_re.size)*100:.2f}% of the values are missing.')
        
        self.X, self.Y = generate_features(pr_re, )
        print(f'{(len(self.Y)/len(pr_re))*100:.2f}% of the values are valid after taking calculating the averages')

        if stats is None:
            print('Calculating stats')
            stats = calculate_stats(self.X, self.Y)
        self.stats = stats

    def inverse_pr_y(self, y, key='y'):
        return y*self.stats[key]['std'] + self.stats[key]['mean']
        
    def apply_stats(self, data, key='x'):
        return (data - self.stats[key]['mean'])/self.stats[key]['std']

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.apply_stats(self.X[index]), self.apply_stats(self.Y[index], key='y') 


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


@jax.jit
def jax_batch(batch):
    x = jnp.stack([b[0] for b in batch]).astype(jnp.float32)
    y = jnp.stack([b[1] for b in batch]).astype(jnp.float32)
    return x,y

def get_scale(data, num_valid=10000):
    return data[0:-num_valid].values.std()

def get_datasets(data, num_valid=10000, is_wh=False, **kwargs):

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

    ds_cls = PrecipitationDatasetWH if is_wh else PrecipitationDataset

    train_ds = ds_cls(data_train, **kwargs)
    valid_ds = ds_cls(data_valid, stats=train_ds.stats, **kwargs)
    
    return train_ds, valid_ds

def get_data_loaders(train_ds, valid_ds, bs=128, **kwargs):
    train_dataloader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=jax_batch, **kwargs)
    valid_dataloader = DataLoader(valid_ds, batch_size=bs, shuffle=False, collate_fn=jax_batch, **kwargs)

    return train_dataloader, valid_dataloader