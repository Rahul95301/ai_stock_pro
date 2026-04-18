# AI Stock Advisor Pro

**JECRC — Jaipur Engineering College and Research Centre**
**Semester VI | Python Minor Project 2025-26**
**B.Tech CSE (AI) | Rajasthan Technical University (RTU), Kota**

---

## Project Overview

AI Stock Advisor Pro is a full-stack Python web application that performs real-time, multi-signal stock market analysis for NSE-listed stocks. It combines machine learning, technical analysis, fundamental analysis, and live news sentiment to generate a clear investment recommendation (BUY / WAIT / AVOID) with detailed reasoning.

### Key Highlights
- **No paid API required** — uses yfinance (free) + RSS news feeds (free)
- **3-year historical analysis** with 1-year deep-dive view
- **27 technical features** engineered from raw OHLCV data
- **Real-time news** fetched from Economic Times, MoneyControl, NDTV Profit, LiveMint
- **VADER NLP** scores each headline for sentiment
- **Random Forest** trained on chronological 80/20 split — never shuffled
- **Comprehensive verdict** with 9 analysis categories and detailed reasoning
- **Groww/Zerodha-inspired dark UI** with Google-style search autocomplete

---

## Project Structure

```
ai_stock_pro/
│
├── app.py               ← Flask web server — RUN THIS
├── config.py            ← All parameters and settings
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
│
├── src/
│   ├── __init__.py
│   ├── stock_search.py  ← 100+ NSE stocks with Google-style autocomplete
│   ├── data_loader.py   ← yfinance price data + RSS news fetching
│   ├── sentiment.py     ← VADER scoring of real news headlines
│   ├── features.py      ← 27 technical indicator features
│   └── model.py         ← Random Forest + comprehensive verdict engine
│
├── templates/
│   └── index.html       ← Full Groww-style dark UI (single file)
│
└── outputs/             ← Auto-created for any saved charts
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Application

```bash
python app.py
```

### 3. Open in Browser

```
http://localhost:5000
```

---

## How It Works

### Data Collection
1. **Live Price** — yfinance fetches current price, volume, market cap, 52-week range
2. **3-Year History** — Daily OHLCV data downloaded from Yahoo Finance (free)
3. **Real News** — RSS feeds from 4 major Indian financial publications fetched at runtime
4. **Fundamentals** — PE, PB, ROE, Debt/Equity, EPS, Revenue growth from yfinance

### Feature Engineering (27 features)
| Category | Features |
|---|---|
| Moving Averages | MA-7, MA-21, MA-50, EMA-12, EMA-26 |
| Momentum | MACD, MACD Signal, MACD Histogram |
| Oscillators | RSI-14, Stochastic %K/%D, Williams %R |
| Volatility | Bollinger Bands (upper/lower/width/%), ATR, ATR Norm |
| Price | Momentum %, Volatility, Price vs MA21, Price vs MA50 |
| Volume | Volume Ratio |
| Sentiment | VADER compound score (from real news) |
| Candle | High-Low Ratio, Open-Close Ratio |

### Machine Learning
- **Algorithm**: Random Forest Classifier (200 trees, max_depth=7)
- **Target**: Will price be higher in 5 trading days? (binary: UP/DOWN)
- **Split**: Chronological 80/20 — no shuffling (time-series correct method)
- **Accuracy**: Typically 55-65% (stock prediction is inherently noisy)

### Verdict Engine (9 Signal Categories)
1. ML Model Prediction (confidence-weighted)
2. RSI — overbought/oversold detection
3. MACD — momentum direction
4. Moving Average alignment (MA-7, MA-21, MA-50)
5. News Sentiment (company-specific weighted 70%, market 30%)
6. Fundamental Analysis (PE, ROE, Debt/Equity, Revenue growth)
7. Historical Returns (1M, 3M, 6M, 1Y trend)
8. 52-Week Position (near high vs near low)
9. Volume Analysis (confirmation signal)

**Verdict outputs**: BULLISH → BUY/ACCUMULATE | MILDLY BULLISH → WAIT & WATCH | BEARISH → AVOID | NEUTRAL → WAIT

---

## Charts Generated
1. **3-Year Price Chart** — Close price + MA-21 + MA-50 + Volume + Sentiment (with 1-year highlight)
2. **1-Year Bollinger Band Chart** — MA-7, MA-21, BB bands
3. **RSI + MACD** — Dual panel with overbought/oversold zones
4. **Multi-Timeframe Returns** — Bar chart for 1M, 3M, 6M, 1Y
5. **Last-Month Signals** — Price with ML prediction markers (▲ UP / ▼ DOWN)
6. **Feature Importance** — Ranked bar chart of RF model features

---

## Technical Decisions

### Why Random Forest?
- Handles non-linear relationships in financial data
- Built-in feature importance ranking
- Robust to outliers and missing data
- No hyperparameter tuning required for good baseline performance

### Why Chronological Split (No Shuffle)?
Shuffling time-series data causes **data leakage** — future data leaks into training. A chronological split simulates real-world deployment where the model predicts on data it has never seen.

### Why 5-Day Forecast?
- 1-day prediction is too noisy (~53% accuracy)
- 5-day smooths out intraday volatility
- Aligns with weekly trading cycles

### Why Free RSS Feeds?
No API key management, no rate limits for academic use, real-time financial news from reputable Indian sources.

---

## Academic Disclaimer

This project is for academic/educational purposes only. It is **not financial advice**. Stock markets involve risk. Always consult a SEBI-registered financial advisor before making investment decisions.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| Flask | Web framework |
| yfinance | Free stock price data API |
| scikit-learn | Random Forest ML model |
| VADER (vaderSentiment) | NLP sentiment analysis |
| Matplotlib | Chart generation |
| Pandas / NumPy | Data processing |
| Requests + XML | RSS feed fetching |
| HTML/CSS/JavaScript | Frontend UI (single-file) |

---

*JECRC · Semester VI · Python Minor Project · 2025-26*
