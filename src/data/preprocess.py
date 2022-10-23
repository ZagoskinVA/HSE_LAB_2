import src.config as cfg
import pandas as pd
import numpy as np
import src.utils



def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    if idx_col in df.columns:
        df = df.set_index(idx_col)
    return df

def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')

    df[cfg.REAL_COLS] = df[cfg.REAL_COLS].astype(np.float32)
    return df

def replace_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in filter(lambda x: x not in cfg.FLOAT_NAN_COLS, cfg.NAN_COLS):
        df[col] = df[col].fillna('None')
    for col in cfg.FLOAT_NAN_COLS:
        df[col] = df[col].fillna(0)
    return df

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.pipe(set_idx, cfg.ID_COL).pipe(replace_nan_values).pipe(cast_types)

def process_target(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.TARGET_COLS] = df[cfg.TARGET_COLS].astype(np.float32)
    return df

        