#%%
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem
import numpy as np

period = 'hourly'
input_fname_tr = Path(__file__).parent / f'train_loss_{period}.csv'
input_fname_val = Path(__file__).parent / f'valid_loss_{period}.csv'
sce = 'train'

#%%
df_tr = pd.read_csv(input_fname_tr)
df_val = pd.read_csv(input_fname_val)

#%%
def extract_variable(df, prefix, postfix, num=5):
    output = []
    for n in range(1, num+1):    
        df_sel = df[f'{prefix}_{n} - {postfix}']
        output.append(df_sel.values)

    output = np.array(output)
    return pd.DataFrame({'mean' : np.mean(output, axis=0), 'error' : sem(output, axis=0)}, index=df.epoch)

#%%

if sce == 'train':
    df_lin = extract_variable(df_tr, f'auck_{period}_lin', 'train_loss_epoch')
    df_nn = extract_variable(df_tr, f'auck_{period}', 'train_loss_epoch')
else:
    df_lin = extract_variable(df_val, f'auck_{period}_lin', 'val_loss')
    df_nn = extract_variable(df_val, f'auck_{period}', 'val_loss')


#%%
def plot_between(df, label):
    ax = sns.lineplot(x=df.index, y=df['mean'].values, label=label, zorder=1)
    lower = df['mean'].values - 2*df['error'].values
    upper = df['mean'].values + 2*df['error'].values
    plt.fill_between(df.index, lower, upper, zorder=0, alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Mean negative log-likelihood")

#%%
plt.grid()
plot_between(df_lin, 'Linear Model')
plot_between(df_nn, 'Neural Network')
plt.xlim(1, 20)

#plt.ylim(0.90, 1.1)
plt.ylim(0.32, 0.42)

plt.savefig(f'auck_loss_{sce}_{period}.png', dpi=300)
plt.show()
# %%
idx = df_lin['mean'].argmin()
df_lin['error'][idx], df_lin['mean'][idx]
# %%
idx = df_nn['mean'].argmin()
df_nn['error'][idx], df_nn['mean'][idx]
#%%
