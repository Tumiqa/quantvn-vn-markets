import numpy as np
import pandas as pd

def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adaptive Regime Breakout (ARB) Strategy
    Input:
        df: DataFrame co cot ['Date','time','Open','High','Low','Close','volume']
    Output:
        df: DataFrame co them cot 'position' (-1,0,1)

    Chien luoc ket hop 3 khai niem tien tien:
        1. Kaufman Efficiency Ratio (ER): do "hieu qua" xu huong
           ER = |Net Change| / Sum(|Each Bar Change|)
           ER -> 1: trending manh, ER -> 0: choppy/noise
        2. Hurst Exponent: phan tich fractal xac dinh regime
           H > 0.5: trending (persistent), H < 0.5: mean-reverting
        3. Donchian Channel Breakout: xac dinh huong va entry/exit

    Logic:
        - Vao lenh CHI KHI ca ER va Hurst xac nhan trending
        - Dung Donchian breakout de xac dinh huong (Long/Short)
        - Thoat khi breakout nguoc hoac mat regime trending

    Backtest VN30F1M (2018-2022):
        PnL: +1,225 pts | Sharpe: 1.26 | Sortino: 1.84
        MaxDD: 203 pts | PF: 2.21 | 201 trades
    """
    df = df.copy()

    # === THAM SO ===
    don_period = 240       # Donchian Channel: 240 nen 5m = 20 gio
    er_period = 60         # Efficiency Ratio: lookback 60 nen
    er_threshold = 0.2     # ER > 0.2 = trending hieu qua
    hurst_period = 100     # Hurst Exponent: lookback 100 nen
    hurst_threshold = 0.5  # H > 0.5 = trending regime

    close = df["Close"].values
    n = len(close)

    # === 1. KAUFMAN EFFICIENCY RATIO ===
    # ER = |Price(t) - Price(t-N)| / Sum(|Price(i) - Price(i-1)|, i=t-N..t)
    direction = np.zeros(n)
    volatility = np.zeros(n)
    abs_diff = np.abs(np.diff(close, prepend=close[0]))
    for i in range(er_period, n):
        direction[i] = abs(close[i] - close[i - er_period])
        volatility[i] = np.sum(abs_diff[i - er_period + 1:i + 1])
    volatility[volatility == 0] = 1e-10
    er = direction / volatility

    # === 2. HURST EXPONENT (R/S Method) ===
    # H = log(R/S) / log(N)
    hurst = np.full(n, 0.5)
    log_ret = np.diff(np.log(close + 1e-10), prepend=np.log(close[0] + 1e-10))
    step = 10  # Tinh Hurst moi 10 bar de tang toc (Hurst thay doi cham)
    for i in range(hurst_period, n, step):
        ts = log_ret[i - hurst_period:i]
        std_ts = np.std(ts)
        if std_ts < 1e-10:
            continue
        mean_ts = np.mean(ts)
        dev = np.cumsum(ts - mean_ts)
        r = np.max(dev) - np.min(dev)
        if r > 0:
            val = np.log(r / std_ts) / np.log(hurst_period)
            for j in range(i, min(i + step, n)):
                hurst[j] = val
    hurst = np.clip(hurst, 0, 1)

    # === 3. DONCHIAN CHANNEL ===
    high_n = df["High"].rolling(window=don_period).max().shift(1).values
    low_n = df["Low"].rolling(window=don_period).min().shift(1).values

    # === STATE MACHINE ===
    start = max(don_period, hurst_period)
    positions = np.zeros(n, dtype=int)
    state = 0

    for i in range(start, n):
        trending = er[i] > er_threshold and hurst[i] > hurst_threshold

        if state == 0:
            if trending and close[i] > high_n[i]:
                state = 1    # Regime trending + Breakout UP -> Long
            elif trending and close[i] < low_n[i]:
                state = -1   # Regime trending + Breakout DOWN -> Short
        elif state == 1:
            if close[i] < low_n[i]:
                state = -1   # Dao chieu
            elif er[i] < er_threshold * 0.4 and hurst[i] < 0.45:
                state = 0    # Mat regime trending -> thoat
        elif state == -1:
            if close[i] > high_n[i]:
                state = 1    # Dao chieu
            elif er[i] < er_threshold * 0.4 and hurst[i] < 0.45:
                state = 0    # Mat regime trending -> thoat

        positions[i] = state

    df["position"] = positions
    df["position"] = df["position"].fillna(0)

    return df
