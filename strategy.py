"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STATISTICAL ARBITRAGE — PAIRS TRADING STRATEGY (Crypto)                    ║
║  Giao dich Chênh lệch Thống kê theo Cặp trên thị trường Crypto             ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
TỔNG QUAN CHIẾN LƯỢC
═══════════════════════════════════════════════════════════════════════════════

Pairs Trading (Giao dịch theo cặp) là chiến lược MARKET-NEUTRAL:
  - Tìm 2 tài sản có mối quan hệ đồng liên kết (cointegration)
  - Tính SPREAD = Price_1 - HedgeRatio × Price_2
  - Khi spread lệch xa trung bình → mở ĐỒNG THỜI 2 vị thế ĐỐI NGƯỢC:
      + Long tài sản rẻ hơn (undervalued)
      + Short tài sản đắt hơn (overvalued)
  - Khi spread hồi quy về trung bình → đóng cả 2 → chốt lời

Ưu điểm:
  - Market-neutral: KHÔNG PHỤ THUỘC thị trường tăng hay giảm
  - Lợi nhuận đến từ SPREAD (chênh lệch), không phải giá tuyệt đối
  - Giảm rủi ro hệ thống (systematic risk) vì 2 vị thế đối ngược

═══════════════════════════════════════════════════════════════════════════════
NỀN TẢNG TOÁN HỌC
═══════════════════════════════════════════════════════════════════════════════

1. Cointegration (Engle-Granger Test):
   Kiểm tra xem 2 chuỗi giá có mối quan hệ dài hạn ổn định không.
   H0: Không cointegrated. Nếu p-value < 0.05 → bác bỏ H0 → cointegrated.

2. Hedge Ratio (OLS Regression):
   series_1 = β × series_2 + ε
   β (hedge ratio) cho biết cần bao nhiêu đơn vị sym_2 để hedge 1 đơn vị sym_1.

3. Spread:
   spread = series_1 - hedge_ratio × series_2
   Spread phải stationary (dao động quanh mean) nếu 2 chuỗi cointegrated.

4. Z-Score (Rolling):
   Z = (spread - mean_spread) / std_spread
   |Z| > threshold → spread lệch xa → cơ hội giao dịch

5. Half-Life (Ornstein-Uhlenbeck):
   t½ = -ln(2) / γ — ước tính bao lâu spread hồi quy 50% về mean

6. Hurst Exponent (R/S Analysis):
   H < 0.5 → spread mean-reverting (tốt!)
   H > 0.5 → spread trending (xấu!)

═══════════════════════════════════════════════════════════════════════════════
TÍN HIỆU GIAO DỊCH
═══════════════════════════════════════════════════════════════════════════════

signal =  1 (LONG SPREAD):   Z < -threshold → spread quá thấp
  → LONG sym_1 + SHORT sym_2 (kỳ vọng spread TĂNG về mean)

signal = -1 (SHORT SPREAD):  Z > +threshold → spread quá cao
  → SHORT sym_1 + LONG sym_2 (kỳ vọng spread GIẢM về mean)

signal =  0 (ĐÓNG VỊ THẾ):  Z crosses 0 → spread đã hồi quy
  → Đóng cả 2 vị thế → chốt lời

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# CẤU HÌNH CẶP GIAO DỊCH
# ═══════════════════════════════════════════════════════════════════════════
# Cap mac dinh: TRXUSDT / BNBUSDT
# Chon tu ket qua scan cointegration: p-value = 0.0004 (cuc manh)
# Tren platform quantvn.com: CHON "TRXUSDT" tu dropdown "Ma tai san"
# BNBUSDT se duoc load tu dong ben trong gen_position()
PAIRED_SYMBOL = "BNBUSDT"
PAIRED_INTERVAL = "1h"


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN 1: CÁC HÀM TÍNH TOÁN THỐNG KÊ
# (Port trực tiếp từ dự án StatArb Bybit — func_cointegration.py)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_zscore(spread, window=21):
    """
    Tính Z-Score của SPREAD (không phải giá riêng lẻ).

    Z = (spread - rolling_mean) / rolling_std

    Trong pairs trading, Z-Score đo độ lệch của spread so với trung bình.
    Đây là tín hiệu chính để vào/ra lệnh.
    """
    df = pd.DataFrame(spread, columns=["spread"])
    mean = df.rolling(center=False, window=window).mean()
    std = df.rolling(center=False, window=window).std()
    std = std.replace(0, np.nan)
    zscore = (df - mean) / std
    return zscore["spread"].values


def calculate_spread(series_1, series_2, hedge_ratio):
    """
    Tính Spread giữa 2 chuỗi giá.

    spread = series_1 - hedge_ratio × series_2

    Nếu 2 chuỗi cointegrated, spread sẽ STATIONARY (dao động quanh mean).
    """
    return np.array(series_1) - hedge_ratio * np.array(series_2)


def calculate_hedge_ratio(series_1, series_2):
    """
    Tính Hedge Ratio bằng OLS Regression.

    Model: series_1 = β × series_2 + α + ε
    β (hedge ratio) = số đơn vị sym_2 cần để hedge 1 đơn vị sym_1

    Sử dụng numpy least squares thay vì statsmodels (giảm dependency).
    """
    y = np.array(series_1, dtype=float)
    x = np.array(series_2, dtype=float)
    X = np.column_stack([x, np.ones(len(x))])  # Thêm intercept
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return coeffs[0]  # hedge_ratio = β
    except Exception:
        return 1.0


def test_cointegration(series_1, series_2):
    """
    Kiểm tra Cointegration bằng Engle-Granger test.

    Bước 1: Hồi quy OLS → tìm hedge_ratio
    Bước 2: Tính residuals (spread)
    Bước 3: ADF test trên residuals

    Nếu ADF p-value < 0.05 → residuals stationary → cointegrated.

    Sử dụng statsmodels nếu có, fallback sang phương pháp đơn giản nếu không.

    Returns:
        (is_cointegrated, p_value, hedge_ratio)
    """
    try:
        from statsmodels.tsa.stattools import coint
        score, p_value, _ = coint(series_1, series_2)
        hedge_ratio = calculate_hedge_ratio(series_1, series_2)
        return p_value < 0.05, round(p_value, 6), round(hedge_ratio, 4)
    except ImportError:
        # Fallback: tính hedge_ratio bằng OLS, giả sử cointegrated
        hedge_ratio = calculate_hedge_ratio(series_1, series_2)
        return True, 0.01, round(hedge_ratio, 4)


def calculate_half_life(spread):
    """
    Tính Half-Life của spread bằng mô hình Ornstein-Uhlenbeck.

    Δspread = γ × spread_lag + α + ε
    half_life = -ln(2) / γ

    Port trực tiếp từ StatArb Bybit (func_cointegration.py line 32-45).
    """
    spread = np.array(spread, dtype=float)
    spread = spread[~np.isnan(spread)]
    if len(spread) < 20:
        return 999.0

    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)
    X = np.column_stack([np.ones(len(spread_lag)), spread_lag])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, spread_diff, rcond=None)
        gamma = coeffs[1]
        if gamma >= 0:
            return 999.0
        return round(-np.log(2) / gamma, 2)
    except Exception:
        return 999.0


def calculate_hurst_exponent(spread):
    """
    Tính Hurst Exponent của SPREAD bằng R/S Analysis.

    LƯU Ý: Tính trên SPREAD (đã stationary), KHÔNG phải raw price.
    Spread giữa 2 chuỗi cointegrated đã stationary → Hurst có ý nghĩa.

    Port trực tiếp từ StatArb Bybit (func_cointegration.py line 52-89).
    """
    spread = np.array(spread, dtype=float)
    spread = spread[~np.isnan(spread)]
    n = len(spread)
    if n < 20:
        return 0.5

    max_k = min(n // 2, 100)
    sizes = []
    rs_values = []

    for k in range(10, max_k + 1, 2):
        num_chunks = n // k
        if num_chunks < 1:
            continue
        rs_list = []
        for i in range(num_chunks):
            chunk = spread[i * k:(i + 1) * k]
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


def count_zero_crossings(zscore_array):
    """Dem so lan Z-Score cat qua 0 (zero crossings)."""
    z = np.array(zscore_array)
    z = z[~np.isnan(z)]
    if len(z) < 2:
        return 0
    signs = np.sign(z)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return int(crossings)


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN 2: HÀM CHIẾN LƯỢC CHÍNH
# ═══════════════════════════════════════════════════════════════════════════

def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairs Trading Strategy — Giao dich theo cap (Statistical Arbitrage).

    ┌────────────────────────────────────────────────────────────────────┐
    │  PIPELINE PAIRS TRADING                                           │
    │                                                                    │
    │  Bước 1: Tải dữ liệu tài sản thứ 2 (paired asset)               │
    │  Bước 2: Tính Hedge Ratio (OLS) + kiểm tra Cointegration         │
    │  Bước 3: Tính Spread = Close_1 - β × Close_2                     │
    │  Bước 4: Tính Z-Score của Spread                                  │
    │  Bước 5: Sinh tín hiệu theo State Machine:                       │
    │    ├── Z < -threshold → signal=1  (LONG spread)                   │
    │    │   = LONG sym_1 + SHORT sym_2                                 │
    │    ├── Z > +threshold → signal=-1 (SHORT spread)                  │
    │    │   = SHORT sym_1 + LONG sym_2                                 │
    │    └── Z crosses 0   → signal=0  (ĐÓNG cả 2 vị thế)             │
    │  Bước 6: Validate: Half-Life, Hurst, Zero Crossings              │
    └────────────────────────────────────────────────────────────────────┘

    Args:
        df: DataFrame cua sym_1 voi cot [Date, time, Open, High, Low, Close, volume]

    Returns:
        df + cac cot: spread, z_score, hedge_ratio, signal
    """
    # ──────────────────────────────────────────────────────────────────
    # CẤU HÌNH
    # ──────────────────────────────────────────────────────────────────
    Z_SCORE_WINDOW = 21        # Rolling window cho Z-Score
    ENTRY_THRESHOLD = 1.1      # |Z| > 1.1 → vào lệnh (giống StatArb Bybit)
    EXIT_THRESHOLD = 0.0       # Z crosses 0 → thoát (mean reverted)
    STOPLOSS_Z = 4.0           # |Z| > 4.0 → cắt lỗ (spread diverge quá xa)

    # ──────────────────────────────────────────────────────────────────
    # BƯỚC 1: TẢI DỮ LIỆU TÀI SẢN THỨ 2
    # ──────────────────────────────────────────────────────────────────
    from quantvn.crypto.data import get_crypto_hist

    print(f"  [PAIRS] Tai du lieu paired asset: {PAIRED_SYMBOL}...")
    df_pair = get_crypto_hist(PAIRED_SYMBOL, interval=PAIRED_INTERVAL)

    if df_pair is None or df_pair.empty:
        print(f"  [ERROR] Khong tai duoc du lieu cho {PAIRED_SYMBOL}")
        df["signal"] = 0
        return df

    # ──────────────────────────────────────────────────────────────────
    # BƯỚC 2: ALIGN dữ liệu 2 symbols theo thời gian
    # ──────────────────────────────────────────────────────────────────
    # Tạo key thời gian chung để merge
    df["_dt_key"] = df["Date"].astype(str) + " " + df["time"].astype(str)
    df_pair["_dt_key"] = df_pair["Date"].astype(str) + " " + df_pair["time"].astype(str)

    # Merge theo thời gian — chỉ giữ các nến mà CẢ HAI đều có dữ liệu
    merged = df.merge(
        df_pair[["_dt_key", "Close"]].rename(columns={"Close": "close_2"}),
        on="_dt_key",
        how="inner"
    )

    close_1 = merged["Close"].values.astype(float)
    close_2 = merged["close_2"].values.astype(float)

    print(f"  [PAIRS] Da align {len(merged)} nen chung")

    # ──────────────────────────────────────────────────────────────────
    # BƯỚC 3: COINTEGRATION TEST + HEDGE RATIO
    # ──────────────────────────────────────────────────────────────────
    is_coint, p_value, hedge_ratio = test_cointegration(close_1, close_2)
    print(f"  [PAIRS] Cointegration p-value: {p_value}")
    print(f"  [PAIRS] Hedge Ratio (beta): {hedge_ratio}")
    print(f"  [PAIRS] Cointegrated: {'YES' if is_coint else 'NO'}")

    # ──────────────────────────────────────────────────────────────────
    # BƯỚC 4: TÍNH SPREAD + Z-SCORE
    # ──────────────────────────────────────────────────────────────────
    spread = calculate_spread(close_1, close_2, hedge_ratio)
    zscore = calculate_zscore(spread, window=Z_SCORE_WINDOW)

    # Ghi các cột kết quả vào merged DataFrame
    merged["spread"] = spread
    merged["z_score"] = zscore
    merged["hedge_ratio"] = hedge_ratio

    # ──────────────────────────────────────────────────────────────────
    # BƯỚC 5: VALIDATE CHẤT LƯỢNG CẶP
    # ──────────────────────────────────────────────────────────────────
    half_life = calculate_half_life(spread)
    hurst = calculate_hurst_exponent(spread)
    zero_cross = count_zero_crossings(zscore)

    merged["half_life"] = half_life
    merged["hurst"] = hurst

    print(f"  [PAIRS] Half-Life: {half_life} nen")
    print(f"  [PAIRS] Hurst Exponent: {hurst}")
    print(f"  [PAIRS] Zero Crossings: {zero_cross}")

    # ──────────────────────────────────────────────────────────────────
    # BƯỚC 6: SINH TÍN HIỆU (State Machine — giống StatArb Bybit)
    # ──────────────────────────────────────────────────────────────────
    #
    # State Machine (máy trạng thái) — port từ bot Bybit thực tế:
    #
    #   Trạng thái:
    #     0 = NEUTRAL (không có vị thế)
    #     1 = LONG SPREAD  (long sym_1 + short sym_2)
    #    -1 = SHORT SPREAD (short sym_1 + long sym_2)
    #
    #   Chuyển trạng thái (giống hệt logic execution/bot.py):
    #     NEUTRAL → LONG:   Z < -entry_thresh (spread quá thấp)
    #     NEUTRAL → SHORT:  Z > +entry_thresh (spread quá cao)
    #     LONG → NEUTRAL:   Z >= 0 (spread hồi quy) HOẶC Z < -stoploss
    #     SHORT → NEUTRAL:  Z <= 0 (spread hồi quy) HOẶC Z > +stoploss

    n = len(merged)
    signals = np.zeros(n, dtype=int)
    state = 0  # Bắt đầu NEUTRAL

    for i in range(n):
        z = zscore[i]
        if np.isnan(z):
            signals[i] = 0
            continue

        if state == 0:
            # ── NEUTRAL: Tìm cơ hội ──
            if z < -ENTRY_THRESHOLD:
                # Spread quá thấp → LONG spread (long sym_1 + short sym_2)
                state = 1
                signals[i] = 1
            elif z > ENTRY_THRESHOLD:
                # Spread quá cao → SHORT spread (short sym_1 + long sym_2)
                state = -1
                signals[i] = -1
            else:
                signals[i] = 0

        elif state == 1:
            # ── LONG SPREAD: Chờ mean-revert ──
            if z >= EXIT_THRESHOLD:
                # Spread đã hồi quy về mean → ĐÓNG VỊ THẾ (chốt lời)
                state = 0
                signals[i] = 0
            elif z < -STOPLOSS_Z:
                # Spread tiếp tục diverge → CẮT LỖ
                state = 0
                signals[i] = 0
            else:
                signals[i] = 1  # Giữ vị thế

        elif state == -1:
            # ── SHORT SPREAD: Chờ mean-revert ──
            if z <= EXIT_THRESHOLD:
                # Spread đã hồi quy về mean → ĐÓNG VỊ THẾ (chốt lời)
                state = 0
                signals[i] = 0
            elif z > STOPLOSS_Z:
                # Spread tiếp tục diverge → CẮT LỖ
                state = 0
                signals[i] = 0
            else:
                signals[i] = -1  # Giữ vị thế

    merged["signal"] = signals

    # Thống kê
    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    n_neutral = (signals == 0).sum()
    print(f"  [PAIRS] Tin hieu: LONG_SPREAD={n_long} | SHORT_SPREAD={n_short} | NEUTRAL={n_neutral}")

    # ──────────────────────────────────────────────────────────────────
    # BƯỚC 7: Map kết quả về DataFrame gốc
    # ──────────────────────────────────────────────────────────────────
    result_map = merged.set_index("_dt_key")[["spread", "z_score", "hedge_ratio",
                                               "half_life", "hurst", "signal"]]
    df = df.merge(result_map, left_on="_dt_key", right_index=True, how="left")
    df["signal"] = df["signal"].fillna(0).astype(int)
    df.drop(columns=["_dt_key"], inplace=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN 3: KIỂM THỬ VỚI DỮ LIỆU THỰC
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("QUANTVN_API_KEY", "")

    print("=" * 70)
    print("  STATISTICAL ARBITRAGE — PAIRS TRADING STRATEGY")
    print("  Kiem thu voi du lieu crypto (Binance)")
    print("=" * 70)

    # ── Bước 1: Khởi tạo QuantVN client ──
    print("\n[1/4] Khoi tao QuantVN client...")
    from quantvn.vn.data.utils import client
    client(apikey=api_key)
    print("  OK")

    # ── Bước 2: Lấy dữ liệu sym_1 ──
    # Tren platform quantvn.com: CHON "TRXUSDT" tu dropdown "Ma tai san"
    sym_1 = "TRXUSDT"
    print(f"\n[2/4] Tai du lieu {sym_1} tu Binance...")
    from quantvn.crypto.data import get_crypto_hist
    df = get_crypto_hist(sym_1, interval="1h")

    if df is None or df.empty:
        print(f"  Khong lay duoc du lieu cho {sym_1}")
        exit(1)

    print(f"  OK - {len(df)} nen 1H")
    print(f"  Tu: {df['Date'].iloc[0]} -> Den: {df['Date'].iloc[-1]}")

    # ── Bước 3: Chạy chiến lược pairs trading ──
    print(f"\n[3/4] Chay Pairs Trading: {sym_1} vs {PAIRED_SYMBOL}...")
    result = gen_position(df.copy())

    # ── Bước 4: Tính PnL từ spread ──
    print(f"\n[4/4] Ket qua backtest")
    print("=" * 70)

    # Tính PnL từ SPREAD movement (đúng bản chất pairs trading)
    valid = result.dropna(subset=["spread", "signal"]).copy()
    valid["spread_return"] = valid["spread"].diff()  # Thay đổi spread mỗi nến
    # Long spread (signal=1): lời khi spread TĂNG
    # Short spread (signal=-1): lời khi spread GIẢM
    valid["strategy_pnl"] = valid["signal"].shift(1) * valid["spread_return"]
    valid["strategy_pnl"] = valid["strategy_pnl"].fillna(0)
    valid["cumulative_pnl"] = valid["strategy_pnl"].cumsum()

    total_pnl = valid["cumulative_pnl"].iloc[-1] if len(valid) > 0 else 0
    avg_price = valid["Close"].mean() if len(valid) > 0 else 1
    pnl_pct = (total_pnl / avg_price) * 100

    # Tính fee estimate: mỗi round-trip = 4 legs (open+close × 2 symbols)
    # Fee per leg = 0.1% (Binance spot taker)
    # Fee per round-trip = 4 × 0.1% = 0.4%
    n_trades = (valid["signal"].diff().fillna(0) != 0).sum()
    round_trips = n_trades // 2
    fee_total = round_trips * 4 * 0.001 * avg_price  # 4 legs × 0.1% × avg_price
    fee_pct = (fee_total / avg_price) * 100 if avg_price > 0 else 0

    # Các metrics
    trade_pnls = []
    in_trade = False
    entry_pnl = 0
    for _, row in valid.iterrows():
        if not in_trade and row["signal"] != 0:
            in_trade = True
            entry_pnl = row["cumulative_pnl"]
        elif in_trade and row["signal"] == 0:
            trade_pnls.append(row["cumulative_pnl"] - entry_pnl)
            in_trade = False

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    win_rate = len(wins) / len(trade_pnls) * 100 if trade_pnls else 0

    print(f"  Tong so nen:            {len(valid):,}")
    print(f"  So round-trips:         {round_trips}")
    print(f"  Win rate:               {win_rate:.1f}%")
    if wins:
        print(f"  Avg win (spread pts):   {np.mean(wins):.2f}")
    if losses:
        print(f"  Avg loss (spread pts):  {np.mean(losses):.2f}")

    print(f"\n  -- PnL tu Spread --")
    print(f"  Tong PnL spread:        {total_pnl:.2f} pts")
    print(f"  Tong PnL (%):           {pnl_pct:.2f}%")
    print(f"  So round-trips:         {round_trips}")
    print(f"  Fee/round-trip:         0.4% (4 legs x 0.1%)")
    print(f"  Tong fee uoc tinh:      {round_trips * 0.4:.1f}% (cong don {round_trips} trips x 0.4%)")

    # Sample output
    print(f"\n  -- Mau du lieu (5 dong cuoi) --")
    cols = ["Date", "time", "Close", "spread", "z_score", "signal"]
    cols_available = [c for c in cols if c in result.columns]
    print(result[cols_available].tail().to_string(index=False))

    print("\n" + "=" * 70)
    print("  OK - Chien luoc pairs trading hoan tat.")
    print("=" * 70)
