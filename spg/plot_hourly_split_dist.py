#%%
from spg import data_utils
import matplotlib.pyplot as plt
import numpy as np
#%%

data = data_utils.load_data_hourly()


# %%
data = data.resample('H').asfreq() 
data += np.random.uniform(0.0, 0.2, size=len(data))
# %%

2**8/24
#%%
data_subset = data
stds = []
all_hrs = []
for n in range(8):
    hours = 2**(n)
    all_hrs.append(hours)
    
    data_average = data_subset.rolling(2).sum()
    mask = (~np.isnan(data_average.values)) & (data_average.values > 5.1)
    ratios = (data_subset[mask] / data_average[mask]).values
    diff = (data_average[mask] - data_subset[mask]).values

    data_subset = data_average
    stds.append(ratios.std())
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

    ax1.set_title(f'{2**(n)}hr, fraction distribution' )
    ax1.hist(ratios, 20, density=True)
    
    ax2.set_title(f'{2**(n)}hr, mean difference' )
    ax2.hist(diff, 20, density=True)
    
    #plt.savefig(f'mean_difference_{n}.png')
    plt.show()
#%%
plt.plot(np.log(list(reversed(all_hrs))), stds)
plt.show()
#plt.plot(1.0/np.array(stds))
# %%
all_hrs
#%%