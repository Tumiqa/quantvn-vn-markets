import numpy as np
import pandas as pd

def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend Following - Donchian Channel Breakout Strategy
    Input:
        df: DataFrame co cot ['Date','time','Open','High','Low','Close','volume']
    Output:
        df: DataFrame co them cot 'position' (-1,0,1)
    """
    df = df.copy()

    # --- Donchian Channel ---
    period = 240
    high_n = df["High"].rolling(window=period).max().shift(1).values
    low_n = df["Low"].rolling(window=period).min().shift(1).values
    close_vals = df["Close"].values
    n = len(df)
    positions = np.zeros(n, dtype=int)
    state = 0

    for i in range(period, n):
        if state == 0:
            if close_vals[i] > high_n[i]:
                state = 1
            elif close_vals[i] < low_n[i]:
                state = -1
        elif state == 1:
            if close_vals[i] < low_n[i]:
                state = -1
        elif state == -1:
            if close_vals[i] > high_n[i]:
                state = 1
        positions[i] = state

    df["position"] = positions
    df["position"] = df["position"].fillna(0)

    return df
