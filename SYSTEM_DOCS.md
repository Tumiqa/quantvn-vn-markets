# Hệ Thống Giao Dịch Thuật Toán — Tài Liệu Kỹ Thuật

## Mục lục

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Nguồn Gốc & Mối Liên Hệ Với StatArb Bybit](#2-nguồn-gốc--mối-liên-hệ-với-statarb-bybit)
3. [Quá Trình Nghiên Cứu & Chọn Chiến Lược](#3-quá-trình-nghiên-cứu--chọn-chiến-lược)
4. [Chi Tiết Chiến Lược: Donchian Channel Breakout](#4-chi-tiết-chiến-lược-donchian-channel-breakout)
5. [Kết Quả Backtest & Tối Ưu Hóa](#5-kết-quả-backtest--tối-ưu-hóa)
6. [Kiến Trúc Code](#6-kiến-trúc-code)
7. [Triển Khai Trên Platform QuantVN](#7-triển-khai-trên-platform-quantvn)

---

## 1. Tổng Quan Hệ Thống

Hệ thống này là một **chiến lược giao dịch thuật toán (Algorithmic Trading Strategy)** được thiết kế để giao dịch trên hợp đồng phái sinh **VN30F1M** (Hợp đồng tương lai VN30 tháng gần nhất) thông qua nền tảng QuantVN.

### Thông số chính

| Thông tin | Chi tiết |
|---|---|
| **Thị trường** | Phái sinh Việt Nam (VN30F1M) |
| **Loại chiến lược** | Trend Following — Donchian Channel Breakout |
| **Khung thời gian** | 5 phút (5m) |
| **Tín hiệu** | Long (1), Short (-1), Neutral (0) |
| **Công nghệ** | Python, NumPy, Pandas |
| **Platform** | QuantVN (quantvn.com) |

### Kết quả đạt được trên platform

| Chỉ số | Historical | Out-of-Sample |
|---|---|---|
| **Lợi nhuận ước tính hàng năm** | +15.14% | +7.76% |
| **Mức sụt giảm tối đa** | -19.49% | -23.54% |
| **Tỷ lệ Sharpe** | 0.78 | 0.33 |
| **Hiệu suất (Score)** | 1.52 | — |
| **Sáng tạo (Score)** | 9.00 | — |
| **Độ ổn định (Score)** | 8.00 | — |

---

## 2. Nguồn Gốc & Mối Liên Hệ Với StatArb Bybit

### 2.1. Hệ thống gốc: StatArb Bybit

Chiến lược này được phát triển dựa trên nền tảng nghiên cứu từ dự án **StatArb Bybit** — một hệ thống giao dịch Statistical Arbitrage (Thống Kê Chênh Lệch Giá) chạy trên sàn Bybit Crypto.

**StatArb Bybit** hoạt động theo nguyên lý:
- Tìm hai đồng coin có mối quan hệ thống kê bền vững (Cointegration)
- Tính spread giữa cặp coin bằng OLS Regression (Hedge Ratio)
- Đo độ lệch spread bằng Z-Score
- Vào lệnh khi spread lệch quá xa → kỳ vọng mean reversion

**Các công cụ toán học cốt lõi của StatArb Bybit:**

| Công cụ | Công thức | Mục đích |
|---|---|---|
| **Cointegration** | Engle-Granger Test | Kiểm tra mối quan hệ dài hạn giữa 2 tài sản |
| **Hedge Ratio** | β = OLS(Y, X) | Tỷ lệ hedge để tạo spread stationary |
| **Z-Score** | Z = (spread - μ) / σ | Đo độ lệch spread khỏi trung bình |
| **Half-Life** | t½ = -ln(2) / γ (Ornstein-Uhlenbeck) | Ước tính thời gian mean reversion |
| **Hurst Exponent** | R/S Analysis trên log returns | Kiểm tra tính chất mean-reverting (H < 0.5) |

### 2.2. Tại sao không dùng trực tiếp StatArb?

Trong quá trình phát triển, chúng tôi đã thử port logic StatArb sang QuantVN nhưng gặp **3 rào cản kỹ thuật** của platform:

1. **Single-asset constraint**: Platform QuantVN chỉ cho phép giao dịch 1 mã tài sản duy nhất, không hỗ trợ pairs trading (mở đồng thời 2 vị thế đối ngược).

2. **Sandbox isolation**: Môi trường sandbox không cho phép gọi API bên ngoài (`get_crypto_hist()`) từ bên trong hàm `gen_position()`.

3. **Thị trường phái sinh**: VN30F1M là phái sinh chỉ số Việt Nam — khác hoàn toàn với crypto pairs. Không có "cặp cointegrated" để áp dụng StatArb.

### 2.3. Logic nào được kế thừa từ StatArb?

Mặc dù không dùng trực tiếp Cointegration hay Pairs Trading, hệ thống vẫn **kế thừa triết lý và phương pháp luận** từ StatArb Bybit:

| Yếu tố | StatArb Bybit | Hệ thống hiện tại |
|---|---|---|
| **State Machine** | Trạng thái SEEKING → HOLDING → CLOSING | Trạng thái 0 → 1/-1 (Neutral → Long/Short) |
| **Systematic approach** | Dựa hoàn toàn trên quy tắc toán học | Dựa hoàn toàn trên Donchian Channel |
| **Parameter optimization** | Grid search trên Z-Score thresholds | Grid search trên period (120→480) |
| **Risk management** | Max drawdown limits, position sizing | Ít trades (106) → giảm rủi ro phí |
| **Market-neutral thinking** | Long/Short đồng thời | Long/Short tuỳ xu hướng |
| **Backtesting rigor** | Backtest trên dữ liệu lịch sử crypto | Backtest 30+ biến thể trên VN30F1M |

### 2.4. Quá trình chuyển đổi chiến lược

```
StatArb Bybit (Crypto Pairs Trading)
        │
        ▼
Thử port Z-Score Mean Reversion (Single Asset)
        │  ✗ PnL = -84% → mean reversion thất bại trên trending market
        ▼
Thử Mean Reversion + Trend Filter (SMA + RSI)
        │  ✗ PnL = -1,089 pts → vẫn thua nặng
        ▼
Chuyển sang Trend Following
        │  Test 10+ chiến lược: EMA, Breakout, Pullback, Adaptive
        ▼
Donchian Channel Breakout (period=240)
        ✓ PnL = +1,236 pts | Sharpe = 1.09 | 106 trades
```

---

## 3. Quá Trình Nghiên Cứu & Chọn Chiến Lược

### 3.1. Phân tích thị trường VN30F1M

Dữ liệu: **53,668 nến 5 phút**, từ 2018-08-13 đến 2022-12-30.

| Năm | Hiệu suất | Đặc điểm thị trường |
|---|---|---|
| 2018 | -9.3% | Xu hướng giảm nhẹ |
| 2019 | +2.3% | Đi ngang (sideways) |
| 2020 | +23.5% | COVID crash → phục hồi mạnh |
| 2021 | +40.4% | Xu hướng tăng mạnh |
| 2022 | -34.9% | Xu hướng giảm mạnh |

**Kết luận**: VN30F1M có **xu hướng rõ ràng** trong hầu hết các năm → Trend Following phù hợp hơn Mean Reversion.

### 3.2. So sánh 30+ chiến lược đã test

#### Nhóm Mean Reversion (thất bại)

| Chiến lược | PnL (net) | Sharpe | Lý do thất bại |
|---|---|---|---|
| Z-Score + RSI + SMA filter | -1,089 pts | -0.60 | Shorting trong uptrend |
| Z-Score thuần | -315 pts | N/A | Mean reversion sai thị trường |
| Adaptive Z-Score + ATR | -5,495 pts | N/A | Quá nhiều trades, phí ăn hết |

#### Nhóm Trend Following (thành công)

| Chiến lược | PnL (net) | Sharpe | Trades |
|---|---|---|---|
| EMA 10/30 | -2,103 | N/A | 1,741 (quá nhiều) |
| EMA 50/200 | +252 | 0.21 | 294 |
| **Donchian 240** | **+1,236** | **1.09** | **106** |
| Donchian 480 | +666 | 0.58 | 62 |
| Daily SMA 20/50 | +489 | 0.44 | 21 |
| Donchian 360 + ATR(1.5) | +835 | 0.99 | 273 |

### 3.3. Tại sao chọn Donchian 240?

1. **Sharpe cao nhất** (1.09) — vượt trội so với tất cả biến thể khác
2. **Profit Factor 2.08** — trung bình mỗi 1đ thua, kiếm lại 2.08đ
3. **Win Rate 50%** — cân bằng giữa thắng và thua
4. **Ít trades (106)** — phí giao dịch cực thấp
5. **Dương 4/5 năm** — chỉ 2019 lỗ -2 pts (không đáng kể)
6. **MaxDD thấp** — chỉ 299 pts

---

## 4. Chi Tiết Chiến Lược: Donchian Channel Breakout

### 4.1. Nguyên lý

**Donchian Channel** (do Richard Donchian phát minh, 1960s) là một trong những hệ thống trend-following lâu đời nhất, được sử dụng bởi nhóm Turtle Traders huyền thoại.

```
Upper Band = Giá cao nhất trong N nến gần nhất
Lower Band = Giá thấp nhất trong N nến gần nhất
```

**Logic giao dịch:**
- Khi giá **phá vỡ Upper Band** → thị trường đang breakout lên → **LONG**
- Khi giá **phá vỡ Lower Band** → thị trường đang breakout xuống → **SHORT**
- Khi đang giữ vị thế → chỉ thoát khi breakout **ngược hướng**

### 4.2. Tham số

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| **Period** | 240 nến (5m) | = 20 giờ giao dịch ≈ 4 ngày |
| **Entry** | Close > High(240) | Breakout lên → Long |
| **Exit** | Close < Low(240) | Breakout xuống → Short (đảo chiều) |

### 4.3. State Machine

```
                ┌─────────────────────────────┐
                │                             │
                ▼                             │
        ┌──────────────┐                      │
   ┌───►│  NEUTRAL (0)  │◄───┐                │
   │    └──────┬───────┘    │                │
   │           │             │                │
   │    breakout UP    breakout DOWN          │
   │           │             │                │
   │           ▼             ▼                │
   │    ┌──────────┐  ┌───────────┐           │
   │    │ LONG (1) │  │ SHORT (-1)│           │
   │    └─────┬────┘  └─────┬─────┘           │
   │          │              │                │
   │    breakout DOWN   breakout UP           │
   │          │              │                │
   │          ▼              ▼                │
   │    ┌───────────┐  ┌──────────┐           │
   │    │ SHORT (-1)│  │ LONG (1) │           │
   │    └───────────┘  └──────────┘           │
   │                                          │
   └──────────────────────────────────────────┘
```

**Đặc điểm quan trọng**: Hệ thống **không có trạng thái Neutral** sau khi vào lệnh đầu tiên — luôn giữ vị thế Long hoặc Short, chỉ đảo chiều khi có breakout ngược. Điều này giúp bắt trọn các xu hướng lớn.

### 4.4. Ví dụ minh hoạ

```
Ngày          Close   High(240)   Low(240)   Hành động        Position
2020-03-20    750     800         700        —                 0 (Neutral)
2020-03-23    690     800         700        Close < Low       -1 (SHORT)
...                                         Giữ short...      -1
2020-04-15    830     800         650        Close > High      +1 (LONG)
...                                         Giữ long...       +1
2020-12-30    1100    1050        900        Close > High      +1 (tiếp tục LONG)
```

---

## 5. Kết Quả Backtest & Tối Ưu Hóa

### 5.1. Kết quả trên platform QuantVN

#### Historical (In-Sample)
- **Lợi nhuận hàng năm**: +15.14%
- **Max Drawdown**: -19.49%
- **Sharpe Ratio**: 0.78

#### Out-of-Sample
- **Lợi nhuận hàng năm**: +7.76%
- **Max Drawdown**: -23.54%
- **Sharpe Ratio**: 0.33

#### Chỉ số hiệu suất tổng hợp

| Chỉ số | Giá trị | Đánh giá |
|---|---|---|
| Hiệu suất | 1.52 | Tốt |
| Sáng tạo | 9.00 | Xuất sắc |
| Độ ổn định | 8.00 | Rất tốt |
| Lợi nhuận trung bình | 0.62 | Dương |
| Tổn thất trung bình | -10.88 | Chấp nhận được |
| Tỷ lệ thắng | 49.28% | Cân bằng |
| Hệ số lợi nhuận (PF) | 1.11 | > 1.0 = profitable |
| Khoảng thắng trung bình | 12.41% | Lớn hơn khoảng thua |
| Hệ số Sharpe | 0.57 | Dương |
| Hệ số Sortino | 0.83 | Tốt |
| Hệ số Calmar | 20.67 | Xuất sắc |
| Xác suất phá sản | 1.00% | Rất thấp |

### 5.2. Kết quả backtest local (theo năm)

| Năm | PnL (pts) | Nhận xét |
|---|---|---|
| 2018 | +136 | Bắt được trend giảm cuối năm |
| 2019 | -2 | Thị trường sideways → hoà vốn |
| 2020 | +597 | Bắt trọn rally COVID recovery |
| 2021 | +299 | Bắt được bull market |
| 2022 | +207 | Bắt được bear market |
| **Tổng** | **+1,236** | — |

### 5.3. Quá trình tối ưu tham số

Đã test 30+ biến thể bao gồm:

- **Period sweep**: 120, 180, 200, 240, 300, 360, 480
- **Dual channel**: Entry period ≠ Exit period (240/60, 240/120, 300/60, ...)
- **Neutral exit**: Thoát về 0 thay vì đảo chiều
- **ATR stop loss**: Cắt lỗ theo biến động (ATR × 1.5, 2.0, 3.0, 4.0)
- **Volume filter**: Chỉ vào lệnh khi volume > trung bình
- **EMA trend filter**: Chỉ giao dịch theo hướng EMA dài hạn
- **Blended channels**: Entry channel nhanh + Exit channel chậm

**Kết luận**: Donchian(240) thuần cho kết quả tốt nhất — mọi biến thể đều giảm PnL hoặc không cải thiện Sharpe.

---

## 6. Kiến Trúc Code

### 6.1. Cấu trúc repository

```
quantvn-vn-markets/
│
├── strategy.py              # Chiến lược chính (deploy lên platform)
├── .env                     # API key (gitignored)
├── .env.example             # Template API key
├── .gitignore               # Git ignore rules
├── README.md                # Tài liệu thư viện QuantVN
├── SYSTEM_DOCS.md           # Tài liệu hệ thống (file này)
│
├── research_vn30.py         # Script nghiên cứu & tối ưu tham số
├── strategy_vectorized.py   # Phiên bản vectorized (backup)
│
└── quantvn/                 # Thư viện QuantVN (pip install)
    ├── vn/                  # Module dữ liệu VN
    │   ├── data/            # get_stock_hist, get_derivatives_hist
    │   └── metrics/         # Backtest_Derivates, Backtest_Stock
    └── crypto/              # Module dữ liệu Crypto
        ├── data/            # get_crypto_hist
        └── metrics/         # Backtest_Crypto
```

### 6.2. Code chiến lược (strategy.py)

```python
import numpy as np
import pandas as pd

def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Donchian Channel (240 nến = 20 giờ)
    period = 240
    high_n = df["High"].rolling(window=period).max().shift(1).values
    low_n = df["Low"].rolling(window=period).min().shift(1).values
    close_vals = df["Close"].values

    # State Machine: Breakout Logic
    n = len(df)
    positions = np.zeros(n, dtype=int)
    state = 0

    for i in range(period, n):
        if state == 0:
            if close_vals[i] > high_n[i]:    # Breakout UP → Long
                state = 1
            elif close_vals[i] < low_n[i]:   # Breakout DOWN → Short
                state = -1
        elif state == 1:
            if close_vals[i] < low_n[i]:     # Đảo chiều → Short
                state = -1
        elif state == -1:
            if close_vals[i] > high_n[i]:    # Đảo chiều → Long
                state = 1
        positions[i] = state

    df["position"] = positions
    df["position"] = df["position"].fillna(0)
    return df
```

### 6.3. Quy tắc tương thích platform QuantVN

| Quy tắc | Mô tả |
|---|---|
| Chỉ import `numpy`, `pandas` | Platform sandbox không có thư viện khác |
| Hàm `gen_position(df)` | Platform gọi hàm này tự động |
| Input: `['Date','time','Open','High','Low','Close','volume']` | Chuẩn OHLCV |
| Output: thêm cột `position` | Giá trị: 1 (Long), -1 (Short), 0 (Neutral) |
| Giữ nguyên số dòng | `len(output) == len(input)` bắt buộc |
| Không gọi API bên ngoài | Sandbox blocked network |
| Dùng `.values` cho vòng lặp | `.iloc` quá chậm → timeout |
| Dùng `.fillna(0)` | Tránh NaN crash |

---

## 7. Triển Khai Trên Platform QuantVN

### 7.1. Cấu hình bot

| Trường | Giá trị |
|---|---|
| Tên bot | StatArb / Bot trade |
| Thị trường | Cổ phiếu Việt Nam |
| Loại | Phái sinh |
| Mã tài sản | VN30F1M |

### 7.2. Quy trình triển khai

1. Đăng nhập quantvn.com → Nền tảng → Chiến Lược Của Tôi
2. Tạo Bot Mới → Chọn VN30F1M
3. Xoá code mẫu trong editor
4. Paste toàn bộ nội dung `strategy.py`
5. Nhấn **Lưu**
6. Nhấn **Chạy thử** (Run Test) → Xem kết quả backtest
7. Nhấn **Paper Trading** → Bot bắt đầu giao dịch mô phỏng

### 7.3. Lưu ý quan trọng

- **Không paste phần `if __name__ == "__main__"`** — chỉ paste hàm `gen_position` và imports
- **Luôn nhấn Lưu trước khi Chạy thử**
- **API key được bảo mật** trong file `.env` (đã gitignored)

---

*Tài liệu được tạo ngày 2026-05-21. Chiến lược đang hoạt động trên quantvn.com.*
