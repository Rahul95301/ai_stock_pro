# =============================================================================
# src/features.py — Feature Engineering Module
# Computes: MA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR,
#           Williams %R, Momentum, Volatility, Volume features + Target
# =============================================================================

import pandas as pd
import numpy as np

FORECAST_DAYS = 5   # Predict 5-day forward trend

FEATURE_COLS = [
    "MA_7", "MA_21", "MA_50",
    "EMA_12", "EMA_26",
    "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower", "BB_Width", "BB_Pct",
    "Stoch_K", "Stoch_D",
    "Williams_R",
    "ATR", "ATR_Norm",
    "Price_Momentum",
    "Volatility",
    "Volume_Ratio",
    "Price_vs_MA21",
    "Price_vs_MA50",
    "sentiment_compound",
    "High_Low_Ratio",
    "Open_Close_Ratio",
]


def engineer_features(price_df: pd.DataFrame,
                       sentiment_df: pd.DataFrame,
                       forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    """
    Merges price + sentiment, computes all technical features, adds Target column.
    Target = 1 if Close price N days later is higher, else 0.
    """
    print("  → Engineering technical features...")

    df = price_df.copy()
    df = df.join(sentiment_df[["sentiment_compound"]], how="left")
    df["sentiment_compound"] = df["sentiment_compound"].fillna(0)

    # ── Moving Averages ───────────────────────────────────────────────────────
    df["MA_7"]  = df["Close"].rolling(7).mean()
    df["MA_21"] = df["Close"].rolling(21).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    # ── Exponential MAs ───────────────────────────────────────────────────────
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # ── MACD ──────────────────────────────────────────────────────────────────
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI_14"] = (100 - 100 / (1 + rs)).fillna(50)

    # ── Bollinger Bands (20-day) ───────────────────────────────────────────────
    bb_mid         = df["Close"].rolling(20).mean()
    bb_std         = df["Close"].rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid.replace(0, np.nan)
    df["BB_Pct"]   = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)

    # ── Stochastic Oscillator (14-day) ────────────────────────────────────────
    low14  = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = ((df["Close"] - low14) / (high14 - low14).replace(0, np.nan) * 100).fillna(50)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # ── Williams %R ───────────────────────────────────────────────────────────
    df["Williams_R"] = ((high14 - df["Close"]) / (high14 - low14).replace(0, np.nan) * -100).fillna(-50)

    # ── ATR (Average True Range) ──────────────────────────────────────────────
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"]      = tr.rolling(14).mean()
    df["ATR_Norm"] = (df["ATR"] / df["Close"].replace(0, np.nan) * 100).round(4)

    # ── Price Momentum (5-day %) ───────────────────────────────────────────────
    df["Price_Momentum"] = df["Close"].pct_change(5) * 100

    # ── Volatility (14-day rolling std of returns) ────────────────────────────
    df["Volatility"] = df["Close"].pct_change().rolling(14).std() * 100

    # ── Volume Ratio (vs 20-day avg) ──────────────────────────────────────────
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean().replace(0, np.nan)

    # ── Relative price vs MAs ─────────────────────────────────────────────────
    df["Price_vs_MA21"] = (df["Close"] - df["MA_21"]) / df["MA_21"].replace(0, np.nan) * 100
    df["Price_vs_MA50"] = (df["Close"] - df["MA_50"]) / df["MA_50"].replace(0, np.nan) * 100

    # ── Candle-body features ──────────────────────────────────────────────────
    df["High_Low_Ratio"]  = df["High"] / df["Low"].replace(0, np.nan)
    df["Open_Close_Ratio"]= df["Open"] / df["Close"].replace(0, np.nan)

    # ── Target: 1 if price N days later > today ───────────────────────────────
    future_close   = df["Close"].shift(-forecast_days)
    df["Target"]   = (future_close > df["Close"]).astype(int)
    df["Future_Return"] = ((future_close - df["Close"]) / df["Close"]) * 100

    # Drop rows with NaN features or target
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURE_COLS + ["Target"], inplace=True)

    available = [f for f in FEATURE_COLS if f in df.columns]
    print(f"  ✔  {len(df)} rows with {len(available)} features engineered")
    return df


def get_latest_indicators(df: pd.DataFrame) -> dict:
    """Extract the most recent row's indicator values as a clean dict."""
    row = df.iloc[-1]
    return {
        "rsi":          round(float(row.get("RSI_14", 50)), 2),
        "macd":         round(float(row.get("MACD", 0)), 4),
        "macd_signal":  round(float(row.get("MACD_Signal", 0)), 4),
        "macd_hist":    round(float(row.get("MACD_Hist", 0)), 4),
        "ma7":          round(float(row.get("MA_7", 0)), 2),
        "ma21":         round(float(row.get("MA_21", 0)), 2),
        "ma50":         round(float(row.get("MA_50", 0)), 2),
        "ema12":        round(float(row.get("EMA_12", 0)), 2),
        "ema26":        round(float(row.get("EMA_26", 0)), 2),
        "bb_upper":     round(float(row.get("BB_Upper", 0)), 2),
        "bb_lower":     round(float(row.get("BB_Lower", 0)), 2),
        "bb_width":     round(float(row.get("BB_Width", 0)), 4),
        "bb_pct":       round(float(row.get("BB_Pct", 0.5)), 4),
        "stoch_k":      round(float(row.get("Stoch_K", 50)), 2),
        "stoch_d":      round(float(row.get("Stoch_D", 50)), 2),
        "williams_r":   round(float(row.get("Williams_R", -50)), 2),
        "atr":          round(float(row.get("ATR", 0)), 2),
        "atr_norm":     round(float(row.get("ATR_Norm", 0)), 4),
        "momentum":     round(float(row.get("Price_Momentum", 0)), 3),
        "volatility":   round(float(row.get("Volatility", 0)), 4),
        "volume_ratio": round(float(row.get("Volume_Ratio", 1)), 3),
        "price_vs_ma21":round(float(row.get("Price_vs_MA21", 0)), 2),
        "price_vs_ma50":round(float(row.get("Price_vs_MA50", 0)), 2),
        "sentiment":    round(float(row.get("sentiment_compound", 0)), 4),
    }


def interpret_rsi(rsi: float) -> str:
    if rsi >= 80: return "Extremely Overbought 🔴"
    if rsi >= 70: return "Overbought 🔴"
    if rsi >= 60: return "Slightly Overbought 🟡"
    if rsi <= 20: return "Extremely Oversold 🟢"
    if rsi <= 30: return "Oversold 🟢"
    if rsi <= 40: return "Slightly Oversold 🟡"
    return "Neutral Zone ⚪"


def interpret_macd(hist: float) -> str:
    if hist > 0.5:  return "Strong Bullish Momentum 🟢"
    if hist > 0:    return "Bullish Momentum 🟢"
    if hist < -0.5: return "Strong Bearish Momentum 🔴"
    if hist < 0:    return "Bearish Momentum 🔴"
    return "Neutral ⚪"
