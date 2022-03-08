from pathlib import Path
from spg import data_utils

def average(pr, num_hrs=32, freq='H'):
    pr = pr.resample(freq).asfreq()
    pr = pr.rolling(num_hrs).sum()[::num_hrs]
    pr = pr.dropna()
    return pr

if __name__ == '__main__':
    loc = 'tarahills'
    av_hr = 32
    input_file = Path(f'/mnt/temp/projects/otago_uni_marsden/data_keep/spg/station_data_hourly/{loc}.nc')
    
    data = data_utils.load_nc(input_file)
    data = average(data, num_hrs=av_hr)

    data_utils.make_nc(data, input_file.parent  / f'{loc}_{av_hr}.nc', units = f'mm/{av_hr}hr')
