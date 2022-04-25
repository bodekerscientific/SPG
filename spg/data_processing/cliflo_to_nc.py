from spg import data_utils
from pathlib import Path
import pandas as pd

def combine_tsv(paths):
    dfs = [data_utils.load_data(p) for p in paths]
    
    if len(dfs) > 1:
        df = pd.concat(dfs, sort=True)
        df = df[~df.index.duplicated(keep='last')]
        df= df.sort_index()
    else:
        df = dfs[0]
        
    return df

if __name__ == '__main__':
    input_path = Path('/mnt/temp/projects/otago_uni_marsden/data_keep/spg/station_data_raw/')
    output_path = Path('/mnt/temp/projects/otago_uni_marsden/data_keep/spg/station_data/')
    locations = {'dunedin' : ['dunedin_btl_gardens_precip.tsv'],
                 'christchurch' : ['christchurch_gardens_1.tsv', 'christchurch_gardens_2.tsv'],
                 'auckland' : ['auckland_albert_park_1.tsv', 'auckland_albert_park_2.tsv', 'auckland_henderson_north.tsv'],
                 'tauranga' : ['tauranga_aero_1.tsv', 'tauranga_aero_2.tsv']}
    
    for loc, path_names in locations.items():
        df = combine_tsv([input_path / name for name in path_names])
        data_utils.save_nc_tprime(df, output_path / (loc + '.nc'), units = 'mm/day')
