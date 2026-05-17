"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STATISTICAL ARBITRAGE — MEAN REVERSION STRATEGY (Crypto)                   ║
║  Chien luoc Hoi quy Trung binh dua tren Thong ke                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
TONG QUAN CHIEN LUOC
═══════════════════════════════════════════════════════════════════════════════

Chien luoc nay ap dung cac cong cu toan hoc tu Statistical Arbitrage (StatArb)
vao giao dich single-asset cryptocurrency.

Nguyen ly Mean Reversion: Gia tai san co xu huong dao dong quanh gia tri
trung binh. Khi gia lech qua xa khoi trung binh (do bang Z-Score), ta ky
vong gia se quay ve → tao co hoi giao dich.

Pipeline xu ly:
  1. Tinh Z-Score = (Close - SMA) / StdDev → do do lech gia
  2. Kiem tra tinh chat mean-reverting bang Hurst Exponent (H < 0.5)
  3. Uoc tinh thoi gian hoi quy bang Half-Life (Ornstein-Uhlenbeck)
  4. Ket hop SMA trend filter + RSI lam bo loc xac nhan
  5. State Machine: vao lenh khi Z-Score vuot nguong, thoat khi Z ve 0

Tin hieu giao dich:
  signal =  1 (MUA):   Z < -threshold AND uptrend AND RSI < oversold
  signal = -1 (BAN):   Z > +threshold AND downtrend AND RSI > overbought
  signal =  0 (NEUTRAL): Z gan 0 hoac khong co dieu kien phu hop

═══════════════════════════════════════════════════════════════════════════════
NEN TANG TOAN HOC (port tu du an StatArb Bybit)
═══════════════════════════════════════════════════════════════════════════════

1. Z-Score (Rolling):
   Z = (x - mu_rolling) / sigma_rolling
   Do do lech gia hien tai so voi trung binh truot. |Z| > 1.5 la tin hieu.

2. Hurst Exponent (R/S Analysis tren log returns):
   H < 0.5 → Mean-reverting (tot cho chien luoc nay)
   H = 0.5 → Random walk
   H > 0.5 → Trending (khong phu hop)

3. Half-Life (Ornstein-Uhlenbeck tren deviation tu mean):
   t1/2 = -ln(2) / gamma
   Cho biet trung binh bao nhieu nen de gia hoi quy 50% ve trung binh.

4. RSI (Relative Strength Index):
   RSI < 30 → Qua ban (oversold) → co hoi mua
   RSI > 70 → Qua mua (overbought) → co hoi ban

5. SMA Trend Filter (Golden Cross / Death Cross):
   SMA50 > SMA200 → Uptrend → chi cho phep LONG
   SMA50 < SMA200 → Downtrend → chi cho phep SHORT

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# PHAN 1: CAC HAM TINH TOAN THONG KE
# (Port tu du an StatArb Bybit — func_cointegration.py)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_zscore(series, window=21):
    """
    Tinh Z-Score theo rolling window.
    Z = (x - mu) / sigma
    Do do lech gia hien tai so voi trung binh truot.
    """
    rolling_mean = series.rolling(window=window, center=False).mean()
    rolling_std = series.rolling(window=window, center=False).std()
    rolling_std = rolling_std.replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


def calculate_rsi(close, period=14):
    """
    Tinh RSI (Relative Strength Index) theo cong thuc Wilder.
    RSI < 30 → qua ban, RSI > 70 → qua mua.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_half_life(series, window=21):
    """
    Tinh Half-Life bang mo hinh Ornstein-Uhlenbeck.
    Tinh tren DEVIATION tu rolling mean (stationary).
    half_life = -ln(2) / gamma

    Port tu StatArb Bybit (func_cointegration.py).
    """
    rolling_mean = series.rolling(window=window).mean()
    deviation = (series - rolling_mean).dropna()
    values = np.array(deviation, dtype=float)
    if len(values) < 20:
        return 999.0

    spread_lag = values[:-1]
    spread_diff = np.diff(values)
    X = np.column_stack([np.ones(len(spread_lag)), spread_lag])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, spread_diff, rcond=None)
        gamma = coeffs[1]
        if gamma >= 0:
            return 999.0
        return round(-np.log(2) / gamma, 2)
    except Exception:
        return 999.0


def calculate_hurst_exponent(series, min_chunk_size=10):
    """
    Tinh Hurst Exponent bang R/S Analysis tren LOG RETURNS.
    Phai tinh tren returns (stationary), KHONG tinh tren raw price.

    H < 0.5 → Mean-reverting
    H = 0.5 → Random walk
    H > 0.5 → Trending

    Port tu StatArb Bybit (func_cointegration.py).
    """
    prices = np.array(series.dropna(), dtype=float)
    prices = prices[prices > 0]
    if len(prices) < 30:
        return 0.5
    log_returns = np.diff(np.log(prices))
    n = len(log_returns)
    if n < 20:
        return 0.5

    max_k = min(n // 2, 100)
    sizes = []
    rs_values = []

    for k in range(min_chunk_size, max_k + 1, 2):
        num_chunks = n // k
        if num_chunks < 1:
            continue
        rs_list = []
        for i in range(num_chunks):
            chunk = log_returns[i * k:(i + 1) * k]
            mean_chunk = np.mean(chunk)
            deviations = chunk - mean_chunk
            cumulative = np.cumsum(deviations)
            r = np.max(cumulative) - np.min(cumulative)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if len(rs_list) > 0:
            sizes.append(k)
            rs_values.append(np.mean(rs_list))

    if len(sizes) < 3:
        return 0.5

    try:
        poly = np.polyfit(np.log(sizes), np.log(rs_values), 1)
        hurst = round(poly[0], 4)
        return max(0.0, min(1.0, hurst))
    except Exception:
        return 0.5


# ═══════════════════════════════════════════════════════════════════════════
# PHAN 2: HAM CHIEN LUOC CHINH
# ═══════════════════════════════════════════════════════════════════════════

def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean-Reversion Strategy voi Trend Filter.

    Pipeline:
      1. Tinh Z-Score (rolling 21 nen) → do do lech gia
      2. Tinh RSI (14 nen) → xac nhan qua mua/qua ban
      3. Tinh SMA50 vs SMA200 → xac dinh xu huong (trend filter)
      4. Tinh Hurst + Half-Life → danh gia chat luong mean-reversion
      5. State Machine → sinh tin hieu {1, -1, 0}

    Quy tac giao dich:
      MUA (signal=1):  Z < -1.5 AND RSI < 35 AND uptrend (SMA50 > SMA200)
      BAN (signal=-1): Z > +1.5 AND RSI > 65 AND downtrend (SMA50 < SMA200)
      THOAT (signal=0): |Z| < 0.3 (spread da hoi quy ve mean)

    Args:
        df: DataFrame voi cot [Date, time, Open, High, Low, Close, volume]

    Returns:
        df + cac cot: z_score, rsi_14, signal
    """
    # ──────────────────────────────────────────────────────────────────
    # CAU HINH THAM SO
    # ──────────────────────────────────────────────────────────────────
    Z_WINDOW = 21           # Rolling window cho Z-Score
    RSI_PERIOD = 14         # Chu ky RSI
    ENTRY_Z = 1.5           # |Z| > 1.5 → vao lenh
    EXIT_Z = 0.3            # |Z| < 0.3 → thoat lenh (mean reverted)
    STOPLOSS_Z = 3.5        # |Z| > 3.5 → cat lo
    RSI_OVERSOLD = 35       # RSI < 35 → xac nhan qua ban
    RSI_OVERBOUGHT = 65     # RSI > 65 → xac nhan qua mua
    SMA_FAST = 50           # SMA nhanh
    SMA_SLOW = 200          # SMA cham

    # ──────────────────────────────────────────────────────────────────
    # BUOC 1: TINH TOAN CHI BAO
    # ──────────────────────────────────────────────────────────────────

    # Z-Score: chi bao chinh
    df["z_score"] = calculate_zscore(df["Close"], window=Z_WINDOW)

    # RSI: bo loc xac nhan
    df["rsi_14"] = calculate_rsi(df["Close"], period=RSI_PERIOD)

    # Trend Filter: SMA50 vs SMA200 (Golden Cross / Death Cross)
    df["sma_fast"] = df["Close"].rolling(window=SMA_FAST).mean()
    df["sma_slow"] = df["Close"].rolling(window=SMA_SLOW).mean()
    df["is_uptrend"] = (df["sma_fast"] > df["sma_slow"]).astype(int)

    # Hurst Exponent (diagnostic — tinh 1 lan)
    hurst = calculate_hurst_exponent(df["Close"])
    df["hurst"] = hurst

    # Half-Life (diagnostic — tinh 1 lan)
    half_life = calculate_half_life(df["Close"], window=Z_WINDOW)
    df["half_life"] = half_life

    # ──────────────────────────────────────────────────────────────────
    # BUOC 2: SINH TIN HIEU (State Machine + Trend Filter)
    # ──────────────────────────────────────────────────────────────────
    #
    # State Machine (may trang thai) — port tu bot StatArb Bybit:
    #
    #   Trang thai:
    #     0 = NEUTRAL (khong co vi the)
    #     1 = LONG    (dang giu vi the mua)
    #    -1 = SHORT   (dang giu vi the ban)
    #
    #   Chuyen trang thai:
    #     NEUTRAL → LONG:   Z < -entry AND RSI < oversold AND uptrend
    #     NEUTRAL → SHORT:  Z > +entry AND RSI > overbought AND downtrend
    #     LONG → NEUTRAL:   Z > -exit (mean reverted) HOAC Z > stoploss
    #     SHORT → NEUTRAL:  Z < +exit (mean reverted) HOAC Z < -stoploss
    #
    #   Trend Filter ngan giao dich nguoc xu huong:
    #     Chi LONG khi uptrend (SMA50 > SMA200) → mua day trong xu huong tang
    #     Chi SHORT khi downtrend (SMA50 < SMA200) → ban dinh trong xu huong giam

    z_vals = df["z_score"].values
    rsi_vals = df["rsi_14"].values
    trend_vals = df["is_uptrend"].values
    n = len(df)

    signals = np.zeros(n, dtype=int)
    state = 0  # Bat dau NEUTRAL

    for i in range(n):
        z = z_vals[i]
        rsi = rsi_vals[i]
        up = trend_vals[i]

        if np.isnan(z) or np.isnan(rsi) or np.isnan(up):
            signals[i] = 0
            continue

        if state == 0:
            # NEUTRAL: tim co hoi vao lenh
            if z < -ENTRY_Z and rsi < RSI_OVERSOLD and up:
                state = 1
                signals[i] = 1
            elif z > ENTRY_Z and rsi > RSI_OVERBOUGHT and not up:
                state = -1
                signals[i] = -1
            else:
                signals[i] = 0

        elif state == 1:
            # LONG: kiem tra thoat
            if z > STOPLOSS_Z:
                state = 0
                signals[i] = 0
            elif z > -EXIT_Z:
                state = 0
                signals[i] = 0
            else:
                signals[i] = 1

        elif state == -1:
            # SHORT: kiem tra thoat
            if z < -STOPLOSS_Z:
                state = 0
                signals[i] = 0
            elif z < EXIT_Z:
                state = 0
                signals[i] = 0
            else:
                signals[i] = -1

    df["signal"] = signals
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PHAN 3: KIEM THU VOI DU LIEU THUC (Chay: python strategy.py)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("QUANTVN_API_KEY", "")

    print("=" * 70)
    print("  STATISTICAL ARBITRAGE — MEAN REVERSION STRATEGY")
    print("  Kiem thu voi du lieu crypto (Binance)")
    print("=" * 70)

    print("\n[1/4] Khoi tao QuantVN client...")
    from quantvn.vn.data.utils import client
    client(apikey=api_key)
    print("  OK")

    # Tren platform: chon ETHUSDT hoac bat ky ma nao tu dropdown
    sym = "ETHUSDT"
    print(f"\n[2/4] Tai du lieu {sym}...")
    from quantvn.crypto.data import get_crypto_hist
    df = get_crypto_hist(sym, interval="1h")

    if df is None or df.empty:
        print(f"  Khong lay duoc du lieu cho {sym}")
        exit(1)

    print(f"  OK - {len(df)} nen 1H")
    print(f"  Tu: {df['Date'].iloc[0]} -> Den: {df['Date'].iloc[-1]}")

    print(f"\n[3/4] Chay chien luoc tren {sym}...")
    result = gen_position(df.copy())

    print(f"\n[4/4] Ket qua")
    print("=" * 70)

    total = len(result)
    n_buy = (result["signal"] == 1).sum()
    n_sell = (result["signal"] == -1).sum()
    n_neutral = (result["signal"] == 0).sum()

    print(f"  Tong nen:     {total:,}")
    print(f"  MUA (1):      {n_buy:,} ({n_buy/total*100:.1f}%)")
    print(f"  BAN (-1):     {n_sell:,} ({n_sell/total*100:.1f}%)")
    print(f"  NEUTRAL (0):  {n_neutral:,} ({n_neutral/total*100:.1f}%)")
    print(f"  Hurst:        {result['hurst'].iloc[0]:.4f}")
    print(f"  Half-Life:    {result['half_life'].iloc[0]:.1f} nen")

    changes = (result["signal"].diff().fillna(0) != 0).sum()
    print(f"  Doi vi the:   {changes}")

    # PnL
    result["ret"] = result["Close"].pct_change()
    result["strat_ret"] = result["signal"].shift(1) * result["ret"]
    cum = (1 + result["strat_ret"].fillna(0)).cumprod()
    pnl = (cum.iloc[-1] - 1) * 100
    bnh = ((result["Close"].iloc[-1] / result["Close"].iloc[0]) - 1) * 100

    print(f"\n  PnL chien luoc: {pnl:.2f}%")
    print(f"  Buy & Hold:     {bnh:.2f}%")

    print(f"\n  5 dong cuoi:")
    cols = ["Date", "time", "Close", "z_score", "rsi_14", "signal"]
    print(result[cols].tail().to_string(index=False))

    print("\n" + "=" * 70)
    print("  OK - Hoan tat.")
    print("=" * 70)
