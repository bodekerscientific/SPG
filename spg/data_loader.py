from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import jax.numpy as jnp

class PrecipitationDataset(Dataset):
    def __init__(self, pr : pd.Series, rd_thresh=0.1):
        self.Y = pr[1:].values
        pr = pr[:-1].values
        self.X = np.stack([pr, pr > rd_thresh])
        

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[:, index], self.Y[index] 

def jax_batch(batch):
    x = jnp.stack([b[0] for b in batch]).astype(jnp.float32)
    y = jnp.stack([b[1] for b in batch]).astype(jnp.float32)
    return x,y

def get_data_loaders(data, bs=128, num_valid=2000, ds_cls=PrecipitationDataset, **kwargs):
    data_train = data[0:-num_valid]
    data_valid = data[-num_valid:]

    scale = data_train.values.std()
    train_dataloader = DataLoader(ds_cls(data_train/scale, **kwargs), batch_size=bs, shuffle=True, collate_fn=jax_batch)
    valid_dataloader = DataLoader(ds_cls(data_valid/scale, **kwargs), batch_size=bs, shuffle=False, collate_fn=jax_batch)

    return train_dataloader, valid_dataloader