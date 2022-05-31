#%%
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

input_fname_tr = Path(__file__).parent / 'train_loss.csv'
input_fname_val = Path(__file__).parent / 'valid_loss.csv'


#%%
sns.lineplot()
#%%