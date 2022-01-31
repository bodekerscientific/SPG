from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax

from spg import data_utils
from bslibs.regression.datetime_utils import datetimes_to_dec_year

class PrecipitationDataset(Dataset):
    def __init__(self, pr : pd.Series, average_hours=[1, 3, 8, 24, 24*2, 24*6], inc_doy=True, inc_tprime=True, rd_thresh=0.1, freq='H'):
        # self.Y = pr[1:].values
        # pr = pr[:-1].values
        # self.X = np.stack([pr, pr > rd_thresh])
        pr_re = pr.resample(freq).asfreq()
        # print(f' {(pr_re.isna().sum() / pr_re.size)*100:.2f}% of the values are missing.')

        features = []

        # Select y, and remove the last day as there would be no label        
        y = pr_re[1:]
        dts = y.index
        y = y.values

        pr_re = pr_re[:-1]
        pr_re_rd = (pr_re > rd_thresh).astype(np.float32)

        if inc_doy:
            # Mod 1 as we only want the fraction
            doy_frac = datetimes_to_dec_year(dts) % 1.0 

            # Use cos and sin doy to remove the jump at the start of a year
            features.extend([np.cos(2*np.pi*doy_frac), np.sin(2*np.pi*doy_frac)])
        
        for av_hr in average_hours:
            features.append(pr_re.rolling(av_hr).mean().values) 
            features.append(pr_re_rd.rolling(av_hr).mean().values)
        
        if inc_tprime:
            df_magic = data_utils.load_magic()
            features.append(data_utils.get_tprime_for_times(dts, df_magic['ssp245']))

        x = np.stack(np.stack(features), axis=1).astype(np.float32)

        # Remove all the nans        
        mask = ~(np.isnan(x).any(axis=1) | np.isnan(y))

        self.X = x[mask, :]
        self.Y = y[mask]

        print(f'{(mask.sum()/mask.size)*100:.2f}% of the values are valid after taking calculating the averages')


    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index] 

@jax.jit
def jax_batch(batch):
    x = jnp.stack([b[0] for b in batch]).astype(jnp.float32)
    y = jnp.stack([b[1] for b in batch]).astype(jnp.float32)
    return x,y

def get_data_loaders(data, bs=128, num_valid=10000, ds_cls=PrecipitationDataset, **kwargs):
    data_train = data[0:-num_valid]
    data_valid = data[-num_valid:]

    scale = data_train.values.std()
    print(f'Scale: {scale :.3f}')

    train_ds = ds_cls(data_train/scale, **kwargs)

    train_dataloader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=jax_batch, **kwargs)
    valid_dataloader = DataLoader(ds_cls(data_valid/scale, **kwargs), batch_size=bs, shuffle=False, collate_fn=jax_batch, **kwargs)

    return train_dataloader, valid_dataloader, len(train_ds[0][0])