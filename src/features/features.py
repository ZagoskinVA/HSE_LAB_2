import src.config as cfg
import pandas as pd
import numpy as np
from src.utils import compute_sleep_time


def add_ohe_for_ms_zoning(df: pd.DataFrame):
    for value in set(df[cfg.ZONING_COL]):
        df[f'Is_{value}'] = df[cfg.ZONING_COL].apply(lambda x: 1 if x == value else 0)
        df[f'Is_{value}'] = df[f'Is_{value}'].astype(np.int8)
    return df

def add_features(df:pd.DataFrame) -> pd.DataFrame:
    return df.pipe(add_ohe_for_ms_zoning)
