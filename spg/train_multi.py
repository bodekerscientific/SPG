from spg import train_spg
from multiprocessing.pool import Pool 
from functools import partial

if __name__ == '__main__':
    func = partial(train_spg.run, cfg_name='base_daily', model_version='v10')
    locations = ['dunedin', 'christchurch', 'auckland', 'tauranga']
    with Pool(4) as p:
        p.map(func, locations)
     
    