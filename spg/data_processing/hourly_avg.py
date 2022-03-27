from pathlib import Path
from spg import data_utils
import pandas as pd


if __name__ == '__main__':
    loc = 'dunedin_combined'
    
    av_hr = 24
    input_file = Path(f'/mnt/temp/projects/otago_uni_marsden/data_keep/spg/station_data_hourly/{loc}.nc')

    data = data_utils.load_nc(input_file)
    data = data.dropna()
    
    mask = data.groupby(data.index.date).count() == 24
    data = data.groupby(data.index.date).sum()
    data = data[mask]
    data.index = pd.DatetimeIndex(data.index)
    
    data_utils.make_nc(data, input_file.parent  / f'{loc}_daily.nc', units = f'mm/day')
