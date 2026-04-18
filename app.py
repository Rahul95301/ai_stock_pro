# =============================================================================
# app.py — Flask Web Application | AI Stock Advisor Pro
# JECRC | Sem VI Python Minor Project 2025-26
# Run: python app.py → http://localhost:5000
# =============================================================================

import warnings, os, sys, io, base64, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import config
from src.stock_search import search_stocks
from src.data_loader  import (fetch_price_data, fetch_live_price,
                               fetch_fundamentals, fetch_real_news,
                               compute_period_returns)
from src.sentiment    import (score_headlines, compute_news_sentiment_summary,
                               build_sentiment_series)
from src.features     import (engineer_features, get_latest_indicators,
                               interpret_rsi, interpret_macd, FEATURE_COLS)
from src.model        import (train_and_evaluate, predict_next,
                               get_feature_importance, generate_verdict)

app = Flask(__name__)

# ── Chart color palette (Groww/Zerodha dark theme) ────────────────────────────
BG, BG2, BG3 = "#0b0e11", "#131920", "#1c2435"
BORDER       = "#263047"
TEXT2        = "#7a8899"
CYAN         = "#00d4aa"
GREEN        = "#26de81"
RED          = "#ff4d6d"
YELLOW       = "#ffd32a"
PURPLE       = "#a29bfe"
BLUE         = "#4a9eff"


# ─────────────────────────────────────────────────────────────────────────────
# CHART GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def chart_price_full(df: pd.DataFrame, ticker: str, df_1y: pd.DataFrame = None):
    """3-year candlestick-style close price with MA overlays + sentiment."""
    fig = plt.figure(figsize=(12, 5.5), facecolor=BG2)
    gs  = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(BG2)
        for sp in ax.spines.values(): sp.set_color(BORDER)
        ax.tick_params(colors=TEXT2, labelsize=8)

    # Price with fill
    ax1.plot(df.index, df["Close"], color=CYAN, linewidth=1.5, label="Close", zorder=3)
    ax1.fill_between(df.index, df["Close"], df["Close"].min() * 0.98,
                     alpha=0.08, color=CYAN)
    if "MA_21" in df.columns:
        ax1.plot(df.index, df["MA_21"], color=YELLOW, linewidth=0.9,
                 linestyle="--", alpha=0.85, label="MA-21", zorder=2)
    if "MA_50" in df.columns:
        ax1.plot(df.index, df["MA_50"], color=PURPLE, linewidth=0.9,
                 linestyle="--", alpha=0.75, label="MA-50", zorder=2)

    # 1-year highlight box
    if df_1y is not None and len(df_1y):
        ax1.axvspan(df_1y.index[0], df_1y.index[-1], alpha=0.06, color=GREEN, zorder=1)
        ax1.axvline(df_1y.index[0], color=GREEN, linewidth=0.8, linestyle=":", alpha=0.6)

    ax1.set_ylabel("Price (₹)", color=TEXT2, fontsize=9)
    ax1.legend(facecolor=BG3, labelcolor=TEXT2, fontsize=8, loc="upper left",
               framealpha=0.8)
    ax1.set_title(f"{ticker} — 3-Year Price Chart  |  Shaded region = Last 1 Year",
                  color="#e0e8f0", fontsize=10, pad=7, loc="left")

    # Volume bars
    colors = [GREEN if df["Close"].iloc[i] >= df["Open"].iloc[i] else RED
              for i in range(len(df))]
    ax2.bar(df.index, df["Volume"] / 1e6, color=colors, width=1.2, alpha=0.6)
    ax2.set_ylabel("Vol (M)", color=TEXT2, fontsize=8)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    # Sentiment overlay
    if "sentiment_compound" in df.columns:
        sent_c = np.where(df["sentiment_compound"] >= 0, GREEN, RED)
        ax3.bar(df.index, df["sentiment_compound"], color=sent_c, alpha=0.5, width=1.2)
        ax3.axhline(0, color=BORDER, linewidth=0.6, linestyle="--")
        ax3.set_ylabel("Sentiment", color=TEXT2, fontsize=8)
        ax3.set_ylim(-1.2, 1.2)
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.tick_right()

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=20, ha="right")
    fig.tight_layout(pad=0.8)
    return fig_to_b64(fig)


def chart_1year(df_1y: pd.DataFrame, ticker: str):
    """Detailed 1-year chart with Bollinger Bands."""
    fig = plt.figure(figsize=(12, 4.5), facecolor=BG2)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(BG2)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.tick_params(colors=TEXT2, labelsize=8)

    ax.plot(df_1y.index, df_1y["Close"], color=CYAN, linewidth=1.6, label="Close", zorder=3)
    ax.fill_between(df_1y.index, df_1y["Close"], df_1y["Close"].min() * 0.97,
                    alpha=0.1, color=CYAN)

    if "MA_7" in df_1y.columns:
        ax.plot(df_1y.index, df_1y["MA_7"],  color=GREEN,  lw=0.9,
                linestyle="--", alpha=0.8, label="MA-7")
    if "MA_21" in df_1y.columns:
        ax.plot(df_1y.index, df_1y["MA_21"], color=YELLOW, lw=0.9,
                linestyle="--", alpha=0.8, label="MA-21")
    if "BB_Upper" in df_1y.columns and "BB_Lower" in df_1y.columns:
        ax.fill_between(df_1y.index, df_1y["BB_Upper"], df_1y["BB_Lower"],
                        alpha=0.07, color=BLUE, label="BB Bands")
        ax.plot(df_1y.index, df_1y["BB_Upper"], color=BLUE, lw=0.6, alpha=0.5)
        ax.plot(df_1y.index, df_1y["BB_Lower"], color=BLUE, lw=0.6, alpha=0.5)

    ax.set_ylabel("Price (₹)", color=TEXT2, fontsize=9)
    ax.set_title(f"{ticker} — Last 1 Year  |  MA-7 · MA-21 · Bollinger Bands",
                 color="#e0e8f0", fontsize=10, pad=7, loc="left")
    ax.legend(facecolor=BG3, labelcolor=TEXT2, fontsize=8, loc="upper left", framealpha=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
    fig.tight_layout(pad=0.8)
    return fig_to_b64(fig)


def chart_rsi_macd(df: pd.DataFrame, ticker: str):
    """RSI + MACD dual panel chart."""
    fig = plt.figure(figsize=(12, 5), facecolor=BG2)
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    for ax in [ax1, ax2]:
        ax.set_facecolor(BG2)
        for sp in ax.spines.values(): sp.set_color(BORDER)
        ax.tick_params(colors=TEXT2, labelsize=8)

    # RSI
    if "RSI_14" in df.columns:
        ax1.plot(df.index, df["RSI_14"], color=YELLOW, linewidth=1.1)
        ax1.axhline(70, color=RED,   linestyle="--", linewidth=0.8, alpha=0.7)
        ax1.axhline(30, color=GREEN, linestyle="--", linewidth=0.8, alpha=0.7)
        ax1.axhline(50, color=BORDER, linestyle=":", linewidth=0.5)
        ax1.fill_between(df.index, df["RSI_14"], 70,
                         where=(df["RSI_14"] > 70), alpha=0.12, color=RED)
        ax1.fill_between(df.index, df["RSI_14"], 30,
                         where=(df["RSI_14"] < 30), alpha=0.12, color=GREEN)
        ax1.set_ylabel("RSI (14)", color=TEXT2, fontsize=9)
        ax1.set_ylim(0, 100)
        ax1.text(df.index[-1], 72, "Overbought", color=RED,   fontsize=7, alpha=0.7)
        ax1.text(df.index[-1], 28, "Oversold",   color=GREEN, fontsize=7, alpha=0.7,
                 va="top")
    ax1.set_title(f"{ticker} — RSI (14) & MACD", color="#e0e8f0",
                  fontsize=10, pad=7, loc="left")

    # MACD
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        ax2.plot(df.index, df["MACD"],        color=BLUE,   linewidth=1.0, label="MACD")
        ax2.plot(df.index, df["MACD_Signal"], color=RED,    linewidth=1.0, label="Signal")
        if "MACD_Hist" in df.columns:
            hist_c = np.where(df["MACD_Hist"] >= 0, GREEN, RED)
            ax2.bar(df.index, df["MACD_Hist"], color=hist_c, alpha=0.5, width=1.2,
                    label="Histogram")
        ax2.axhline(0, color=BORDER, linewidth=0.7, linestyle="--")
        ax2.set_ylabel("MACD", color=TEXT2, fontsize=9)
        ax2.legend(facecolor=BG3, labelcolor=TEXT2, fontsize=8, loc="upper left",
                   framealpha=0.8)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20, ha="right")
    fig.tight_layout(pad=0.8)
    return fig_to_b64(fig)


def chart_returns_bar(period_returns: dict, ticker: str):
    """Multi-timeframe returns bar chart."""
    labels = ["1 Month", "3 Months", "6 Months", "1 Year"]
    keys   = ["return_1m", "return_3m", "return_6m", "return_1y"]
    values = [period_returns.get(k) or 0 for k in keys]
    colors = [GREEN if v >= 0 else RED for v in values]

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=BG2)
    ax.set_facecolor(BG2)
    bars = ax.bar(labels, values, color=colors, edgecolor="none", width=0.55,
                  alpha=0.85)
    ax.axhline(0, color=BORDER, linewidth=1.0)
    for bar, val in zip(bars, values):
        y = bar.get_height() + (0.4 if val >= 0 else -1.2)
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{val:+.1f}%", ha="center", color="white", fontsize=11,
                fontweight="bold")
    ax.set_ylabel("Return %", color=TEXT2, fontsize=9)
    ax.set_title(f"{ticker} — Multi-Timeframe Returns",
                 color="#e0e8f0", fontsize=10, pad=7, loc="left")
    ax.tick_params(colors=TEXT2, labelsize=9)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    fig.tight_layout(pad=0.8)
    return fig_to_b64(fig)


def chart_feature_importance(feat_imp: list, ticker: str):
    """Horizontal bar chart of top feature importances."""
    top = feat_imp[:14]
    names  = [x["feature"] for x in top]
    values = [x["importance"] for x in top]
    colors = [RED if "sentiment" in n.lower() else
              PURPLE if n.startswith("BB") else
              YELLOW if "MA" in n or "EMA" in n else
              CYAN for n in names]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG2)
    ax.set_facecolor(BG2)
    bars = ax.barh(names, values, color=colors, edgecolor="none", height=0.62, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color=TEXT2, fontsize=8)
    ax.set_xlabel("Importance Score", color=TEXT2, fontsize=9)
    ax.set_title(f"{ticker} — Feature Importance  |  Cyan=Technical  Purple=BB  Yellow=MA  Red=Sentiment",
                 color="#e0e8f0", fontsize=9, pad=7, loc="left")
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.set_xlim(0, max(values) * 1.35)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    fig.tight_layout(pad=0.8)
    return fig_to_b64(fig)


def chart_signals_month(df: pd.DataFrame, ticker: str):
    """Last 1-month price with UP/DOWN signal markers."""
    df_m = df.iloc[-21:].copy()
    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor=BG2)
    ax.set_facecolor(BG2)
    ax.plot(df_m.index, df_m["Close"], color=CYAN, linewidth=1.5, zorder=2)
    ax.fill_between(df_m.index, df_m["Close"], df_m["Close"].min() * 0.99,
                    alpha=0.09, color=CYAN)
    if "Target" in df_m.columns:
        up   = df_m[df_m["Target"] == 1]
        down = df_m[df_m["Target"] == 0]
        ax.scatter(up.index,   up["Close"],   color=GREEN, marker="^",
                   s=55, alpha=0.9, zorder=3, label="Next UP ↑")
        ax.scatter(down.index, down["Close"], color=RED,   marker="v",
                   s=55, alpha=0.9, zorder=3, label="Next DOWN ↓")
        ax.legend(facecolor=BG3, labelcolor=TEXT2, fontsize=9, loc="upper left",
                  framealpha=0.8)
    ax.set_ylabel("Price (₹)", color=TEXT2, fontsize=9)
    ax.set_title(f"{ticker} — Last 1 Month Price with ML Prediction Signals",
                 color="#e0e8f0", fontsize=10, pad=7, loc="left")
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
    for sp in ax.spines.values(): sp.set_color(BORDER)
    fig.tight_layout(pad=0.8)
    return fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if len(q) < 1:
        return jsonify([])
    results = search_stocks(q, max_results=8)
    return jsonify([{"ticker": t, "name": n, "score": s} for t, n, s in results])


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    data   = request.get_json()
    ticker = (data.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    try:
        print(f"\n{'='*55}")
        print(f"  Analysing: {ticker}")
        print(f"{'='*55}")

        # 1. Live price & fundamentals
        live         = fetch_live_price(ticker)
        fundamentals = fetch_fundamentals(ticker)
        company_name = live.get("company", ticker)

        # 2. Real news from RSS feeds
        raw_news     = fetch_real_news(ticker, company_name, max_articles=15)
        scored_news  = score_headlines(raw_news)
        news_summary = compute_news_sentiment_summary(scored_news)
        latest_news_score = news_summary.get("overall_score", 0.0)

        # 3. Historical price data (3 years)
        df_3y = fetch_price_data(ticker, years=3)
        period_returns = compute_period_returns(df_3y)

        # 4. Build sentiment time series (anchored to real news score)
        sentiment_df = build_sentiment_series(df_3y.index, ticker, latest_news_score)

        # 5. Feature engineering
        full_df = engineer_features(df_3y, sentiment_df)

        # 6. 1-year subset (for charts)
        from datetime import datetime, timedelta
        cutoff_1y = df_3y.index[-1] - timedelta(days=365)
        df_1y_chart = full_df[full_df.index >= cutoff_1y]

        # 7. Train model
        available = [f for f in FEATURE_COLS if f in full_df.columns]
        model, scaler, X_test, y_test, y_pred, used_feats = train_and_evaluate(
            full_df, feature_cols=available,
            test_size=config.TEST_SIZE,
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            random_state=config.RANDOM_STATE,
        )
        acc = accuracy_score(y_test, y_pred) * 100

        # 8. Predict
        prediction   = predict_next(model, scaler, full_df, used_feats)
        indicators   = get_latest_indicators(full_df)
        feat_imp     = get_feature_importance(model, used_feats)

        # 9. Generate comprehensive verdict
        verdict = generate_verdict(
            prediction     = prediction,
            indicators     = indicators,
            fundamentals   = fundamentals,
            period_returns = period_returns,
            news_sentiment = news_summary,
            live_price     = live,
            accuracy       = acc,
        )

        # 10. Generate charts
        charts = {
            "price_3y":  chart_price_full(full_df, ticker, df_1y_chart),
            "price_1y":  chart_1year(df_1y_chart, ticker),
            "rsi_macd":  chart_rsi_macd(full_df.iloc[-252:], ticker),
            "returns":   chart_returns_bar(period_returns, ticker),
            "features":  chart_feature_importance(feat_imp, ticker),
            "signals":   chart_signals_month(full_df, ticker),
        }

        return jsonify({
            "ticker":         ticker,
            "company":        company_name,
            "live":           live,
            "fundamentals":   fundamentals,
            "period_returns": period_returns,
            "indicators":     indicators,
            "prediction":     prediction,
            "verdict":        verdict,
            "news_summary":   news_summary,
            "news":           scored_news[:10],
            "accuracy":       round(acc, 2),
            "rows_loaded":    len(full_df),
            "feature_count":  len(used_feats),
            "feat_imp":       feat_imp[:10],
            "charts":         charts,
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n  ERROR: {e}\n{tb}")
        return jsonify({"error": str(e), "trace": tb}), 500


@app.route("/api/chart_data")
def api_chart_data():
    """
    Returns OHLCV + RSI + MACD + Volume as JSON for Lightweight Charts.
    Called after analyse to render interactive charts.
    """
    ticker = request.args.get("ticker", "").strip().upper()
    period = request.args.get("period", "3y")   # 1m, 3m, 6m, 1y, 3y
    if not ticker:
        return jsonify({"error": "No ticker"}), 400

    period_map = {"1m": 30, "3m": 90, "6m": 180, "1y": 365, "3y": 1095}
    days = period_map.get(period, 1095)

    try:
        from datetime import datetime, timedelta
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=days)
        import yfinance as yf
        raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if raw.empty:
            return jsonify({"error": "No data"}), 404

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[["Open","High","Low","Close","Volume"]].copy()
        raw = raw.loc[:, ~raw.columns.duplicated()]
        raw.index = pd.to_datetime(raw.index)
        raw.dropna(inplace=True)

        # Compute RSI
        delta = raw["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        raw["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

        # Compute MACD
        ema12 = raw["Close"].ewm(span=12, adjust=False).mean()
        ema26 = raw["Close"].ewm(span=26, adjust=False).mean()
        raw["MACD"]        = ema12 - ema26
        raw["MACD_Signal"] = raw["MACD"].ewm(span=9, adjust=False).mean()
        raw["MACD_Hist"]   = raw["MACD"] - raw["MACD_Signal"]

        # MA overlays
        raw["MA20"] = raw["Close"].rolling(20).mean()
        raw["MA50"] = raw["Close"].rolling(50).mean()

        # Bollinger Bands
        bb_std = raw["Close"].rolling(20).std()
        raw["BB_Upper"] = raw["MA20"] + 2 * bb_std
        raw["BB_Lower"] = raw["MA20"] - 2 * bb_std

        raw.dropna(inplace=True)

        def ts(dt):
            """Convert datetime index to Unix timestamp (seconds) — required by Lightweight Charts."""
            import calendar
            return int(calendar.timegm(dt.timetuple()))

        candles = []
        volume  = []
        rsi_data    = []
        macd_data   = []
        macd_sig    = []
        macd_hist   = []
        ma20_data   = []
        ma50_data   = []
        bb_upper    = []
        bb_lower    = []

        for dt, row in raw.iterrows():
            t = ts(dt)
            candles.append({"time": t, "open": round(float(row["Open"]),2),
                            "high": round(float(row["High"]),2),
                            "low":  round(float(row["Low"]),2),
                            "close":round(float(row["Close"]),2)})
            volume.append({"time": t, "value": int(row["Volume"]),
                           "color": "rgba(38,222,129,0.4)" if row["Close"] >= row["Open"] else "rgba(255,77,109,0.4)"})
            rsi_data.append({"time": t, "value": round(float(row["RSI"]), 2)})
            macd_data.append({"time": t, "value": round(float(row["MACD"]), 4)})
            macd_sig.append( {"time": t, "value": round(float(row["MACD_Signal"]), 4)})
            macd_hist.append({"time": t, "value": round(float(row["MACD_Hist"]), 4),
                              "color": "rgba(38,222,129,0.6)" if row["MACD_Hist"] >= 0 else "rgba(255,77,109,0.6)"})
            ma20_data.append({"time": t, "value": round(float(row["MA20"]), 2)})
            ma50_data.append({"time": t, "value": round(float(row["MA50"]), 2)})
            bb_upper.append( {"time": t, "value": round(float(row["BB_Upper"]), 2)})
            bb_lower.append( {"time": t, "value": round(float(row["BB_Lower"]), 2)})

        return jsonify({
            "candles":   candles,
            "volume":    volume,
            "rsi":       rsi_data,
            "macd":      macd_data,
            "macd_sig":  macd_sig,
            "macd_hist": macd_hist,
            "ma20":      ma20_data,
            "ma50":      ma50_data,
            "bb_upper":  bb_upper,
            "bb_lower":  bb_lower,
            "ticker":    ticker,
            "period":    period,
            "rows":      len(candles),
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/live")
def api_live():
    """Lightweight endpoint for real-time price polling (every 30s)."""
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "No ticker"}), 400
    try:
        live = fetch_live_price(ticker)
        return jsonify(live)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AI Stock Advisor Pro — JECRC Sem VI Minor Project")
    print("  Real News · 3-Year Analysis · ML Verdict Engine")
    print("=" * 60)
    print("\n  ➜  Open: http://localhost:5000\n")
    app.run(debug=False, port=5000)
