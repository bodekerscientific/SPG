from pathlib import Path
from spg import data_utils

if __name__ == '__main__':
    loc = 'tauranga'
    av_hr = 32
    input_file = Path(f'/mnt/temp/projects/otago_uni_marsden/data_keep/spg/station_data_hourly/{loc}.nc')
    
    data = data_utils.load_nc(input_file)
    data = data_utils.average(data, num_hrs=av_hr)

    data_utils.make_nc(data, input_file.parent  / f'{loc}_{av_hr}.nc', units = f'mm/{av_hr}hr')
