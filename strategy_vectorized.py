import numpy as np
import pandas as pd

def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend Following - Donchian Channel Breakout (Vectorized)
    Input:
        df: DataFrame co cot ['Date','time','Open','High','Low','Close','volume']
    Output:
        df: DataFrame co them cot 'position' (-1,0,1)
    """
    df = df.copy()

    # --- Tham so ---
    period = 240

    # --- Donchian Channel ---
    df["high_n"] = df["High"].rolling(window=period).max().shift(1)
    df["low_n"] = df["Low"].rolling(window=period).min().shift(1)

    # --- Breakout signals (vectorized, khong dung for loop) ---
    df["break_up"] = (df["Close"] > df["high_n"]).astype(int)
    df["break_down"] = (df["Close"] < df["low_n"]).astype(int)

    # --- Position: dung net signal ---
    df["raw_signal"] = df["break_up"] - df["break_down"]
    df["raw_signal"] = df["raw_signal"].replace(0, np.nan)
    df["position"] = df["raw_signal"].ffill().fillna(0).astype(int)

    return df
