#%%
import pandas as pd
from pathlib import Path
from bslibs import env

def subset(df, region='World|Southern Hemisphere|Land'):
    df = df.loc[df['region'] == region]
    scn = list(df['scenario'].values)
    df_out = df.T.iloc[7:]
    df_out.columns = scn
    df_out.index.name = 'date'
    return df_out

if __name__ == '__main__':
    input_p = env.tprojects('otago_uni_marsden/data_keep/spg/MAGICC/magicc_runs_20210630.csv')
    output_p = Path('data/magic_tprime_sh_land.csv')
    
    df = subset(pd.read_csv(input_p))
    df.to_csv(output_p)
    