import pandas as pd
import numpy as np


def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistical Arbitrage - Mean Reversion Strategy (Z-Score).

    Chien luoc hoi quy trung binh: mua khi gia giam qua sau (oversold),
    ban khi gia tang qua cao (overbought), dua tren Z-Score.

    Pipeline:
      1. Tinh Z-Score = (Close - SMA) / StdDev
      2. State Machine sinh position {1, -1, 0}
      3. Trend Filter (SMA50 vs SMA200) ngan giao dich nguoc xu huong
    """
    df = df.copy()

    # --- Tham so ---
    window = 21
    entry_z = 1.5
    exit_z = 0.3
    sma_fast_period = 50
    sma_slow_period = 200

    # --- Tinh Z-Score ---
    rolling_mean = df["Close"].rolling(window=window).mean()
    rolling_std = df["Close"].rolling(window=window).std()
    rolling_std = rolling_std.replace(0, np.nan)
    z_score = (df["Close"] - rolling_mean) / rolling_std
    z_score = z_score.fillna(0)

    # --- Trend Filter ---
    sma_fast = df["Close"].rolling(window=sma_fast_period).mean()
    sma_slow = df["Close"].rolling(window=sma_slow_period).mean()
    is_uptrend = (sma_fast > sma_slow).fillna(False)

    # --- State Machine ---
    z_vals = z_score.values
    trend_vals = is_uptrend.values
    n = len(df)
    positions = np.zeros(n, dtype=int)
    state = 0

    for i in range(n):
        z = z_vals[i]

        if state == 0:
            if z < -entry_z and trend_vals[i]:
                state = 1
            elif z > entry_z and not trend_vals[i]:
                state = -1
        elif state == 1:
            if z > -exit_z:
                state = 0
        elif state == -1:
            if z < exit_z:
                state = 0

        positions[i] = state

    df["position"] = positions

    return df
