import numpy as np
import pandas as pd

def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend Following - Donchian Channel Breakout Strategy
    Input:
        df: DataFrame co cot ['Date','time','Open','High','Low','Close','volume']
    Output:
        df: DataFrame co them cot 'position' (-1,0,1)

    Chien luoc Donchian Channel Breakout:
        - Tinh kenh gia cao nhat/thap nhat trong 240 nen (20 gio)
        - Long khi gia pha vo kenh tren (breakout len)
        - Short khi gia pha vo kenh duoi (breakout xuong)
        - Dao chieu khi breakout nguoc huong

    Backtest VN30F1M (2018-2022):
        PnL: +1,236 pts | Sharpe: 1.09 | MaxDD: 299 pts | 106 trades
    """
    df = df.copy()

    # --- Tham so ---
    period = 240  # 240 nen 5m = 20 gio giao dich (~4 ngay)

    # --- Donchian Channel ---
    df["high_n"] = df["High"].rolling(window=period).max()
    df["low_n"] = df["Low"].rolling(window=period).min()

    # --- State Machine: Breakout Logic ---
    n = len(df)
    positions = np.zeros(n)
    state = 0

    for i in range(period, n):
        prev_high = df["high_n"].iloc[i - 1]
        prev_low = df["low_n"].iloc[i - 1]
        close = df["Close"].iloc[i]

        if state == 0:
            if close > prev_high:
                state = 1   # Breakout len -> Long
            elif close < prev_low:
                state = -1  # Breakout xuong -> Short
        elif state == 1:
            if close < prev_low:
                state = -1  # Dao chieu -> Short
        elif state == -1:
            if close > prev_high:
                state = 1   # Dao chieu -> Long

        positions[i] = state

    df["position"] = positions
    df["position"] = df["position"].fillna(0)

    return df
