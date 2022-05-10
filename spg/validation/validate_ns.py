#%%
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
#%%
target_loc = 'christchurch'
target_epoch = 23

#%%

def plot_reg_fit(x, y, fit, label, scatter_label='Obs', ci_upper=None, ci_lower=None, ax=None, figsize=(12, 8), 
                 label_dict={}, scatter_kwargs={}, line_kwargs={}):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y, label=scatter_label, **scatter_kwargs)
    res = ax.plot(x, fit, "b-", label=f"Fit: {label}", color='black')
    
    
    if ci_upper is not None:
        ax.plot(x, ci_upper, "r--", label='1 Sigmia')
        ax.plot(x, ci_lower, "r--")

    ax.legend(loc="best")
    #ax.set_xlim(x.min(), x.max())
    return ax

def fit_and_plot(x, y, **kwargs):
    X = sm.add_constant(x)

    mod = sm.OLS(y, X)
    res = mod.fit()

    predictions = res.get_prediction()
    sum_df = predictions.summary_frame(alpha=0.05)

    plot_reg_fit(x, y, sum_df['mean'], f'Fit {res.params[1]:.3f}t + {res.params[0]:.3f}',
                       ci_upper=sum_df['mean_ci_upper'], ci_lower=sum_df['mean_ci_lower'], **kwargs)

#%%

input = f'/mnt/temp/projects/otago_uni_marsden/data_keep/spg/station_data/{target_loc}.nc'
ds_target = xr.open_dataset(input).load()
#%%
input_folder = Path(f'/mnt/temp/projects/otago_uni_marsden/data_keep/spg/ensemble_daily/v8/{target_loc}_epoch_{str(target_epoch).zfill(3)}/')
files = list(input_folder.glob(f'{target_loc}_*.nc'))

#%%
ds = xr.open_mfdataset(files, concat_dim = 'ens', combine='nested').load()
#%%
def quantile_plot(ds, q=0.99):
    ds_grp = ds.groupby(ds.time.dt.year)
    data = ds_grp.quantile(q)['precipitation'].values.reshape(-1)
    t_prime = ds_grp.mean()['tprime'].values.reshape(-1)
    
    fit_and_plot(t_prime, data)
    #plt.xlim(0.4, 1.6)
    #plt.show()

# %%
target_q = 0.999
quantile_plot(ds, target_q)
quantile_plot(ds_target, target_q)
# %%
