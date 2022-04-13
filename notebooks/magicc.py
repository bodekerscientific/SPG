from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class Magicc:
    def __init__(self, path):
        magicc_path = Path(path)
        self.df = pd.read_csv(magicc_path, index_col="date", parse_dates=True)

    def create_interpolator(self, pathway):
        def date_to_float(date):
            return ((date - np.datetime64("2000-01-01T00")) / np.timedelta64(1,"D")).astype(float)

        x = date_to_float(self.df.index.to_numpy())
        y = self.df[pathway].to_numpy()
        date_as_float_to_tprime_sh_land = interp1d(
            x = x,
            y = y
        )

        def date_to_tprime_sh_land(date):
            return date_as_float_to_tprime_sh_land(date_to_float(date))

        return date_to_tprime_sh_land
