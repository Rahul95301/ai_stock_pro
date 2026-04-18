# =============================================================================
# src/model.py — ML Model + Comprehensive Analysis Engine
# Random Forest Classifier with chronological split
# Final Verdict Engine: generates human-readable BUY / WAIT / AVOID advice
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def train_and_evaluate(df: pd.DataFrame,
                        feature_cols: list,
                        test_size: float = 0.2,
                        n_estimators: int = 200,
                        max_depth: int = 7,
                        min_samples_leaf: int = 5,
                        min_samples_split: int = 12,
                        random_state: int = 42):
    """
    Chronological train/test split → train Random Forest → return model + results.
    IMPORTANT: Never shuffle time-series data.
    """
    print("  → Training Random Forest Classifier...")

    available = [f for f in feature_cols if f in df.columns]
    X = df[available]
    y = df["Target"]

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  ✔  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        min_samples_leaf  = min_samples_leaf,
        min_samples_split = min_samples_split,
        random_state      = random_state,
        n_jobs            = -1,
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc    = accuracy_score(y_test, y_pred)

    print(f"  ✔  Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"],
                                 zero_division=0))
    return model, scaler, X_test, y_test, y_pred, available


def predict_next(model, scaler, df: pd.DataFrame, feature_cols: list) -> dict:
    """Predict trend for next N days using latest data row."""
    available = [f for f in feature_cols if f in df.columns]
    latest    = df[available].iloc[-1:].copy()
    latest_s  = scaler.transform(latest)
    pred      = int(model.predict(latest_s)[0])
    proba     = model.predict_proba(latest_s)[0]
    prob_up   = round(float(proba[1]) * 100, 1)
    prob_down = round(float(proba[0]) * 100, 1)

    return {
        "prediction": pred,
        "prob_up":    prob_up,
        "prob_down":  prob_down,
        "label":      "📈 UP" if pred == 1 else "📉 DOWN",
        "confidence": max(prob_up, prob_down),
    }


def get_feature_importance(model, feature_cols: list) -> list:
    """Returns sorted feature importance list."""
    importance = model.feature_importances_
    pairs = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    return [{"feature": f, "importance": round(float(v), 5)} for f, v in pairs]


# ── Comprehensive Verdict Engine ───────────────────────────────────────────────

def generate_verdict(
    prediction: dict,
    indicators: dict,
    fundamentals: dict,
    period_returns: dict,
    news_sentiment: dict,
    live_price: dict,
    accuracy: float,
) -> dict:
    """
    Synthesises ALL signals into a final investment verdict.
    
    Considers:
    - ML model prediction (RF classifier, 5-day)
    - Technical indicators (RSI, MACD, Bollinger, Stochastic)
    - Fundamental data (PE, ROE, Debt/Equity, Revenue growth)
    - News sentiment (VADER-scored real RSS headlines)
    - Historical returns (1M, 3M, 6M, 1Y)
    - 52-week position
    
    Returns: verdict dict with signal, score, reasons[], action, description
    """

    bull_points = 0
    bear_points = 0
    reasons     = []
    signals     = {}

    # ── 1. ML Prediction ──────────────────────────────────────────────────────
    prob_up   = prediction.get("prob_up", 50)
    prob_down = prediction.get("prob_down", 50)
    pred_val  = prediction.get("prediction", 0)
    confidence= prediction.get("confidence", 50)

    if pred_val == 1:
        weight = 2 if confidence >= 65 else 1
        bull_points += weight
        reasons.append({
            "category": "ML Model",
            "signal": "bullish",
            "text": f"Random Forest model predicts UP trend with {prob_up}% probability over next 5 trading days (model accuracy: {accuracy:.1f}%)",
            "icon": "🤖"
        })
    else:
        weight = 2 if confidence >= 65 else 1
        bear_points += weight
        reasons.append({
            "category": "ML Model",
            "signal": "bearish",
            "text": f"Random Forest model predicts DOWN trend with {prob_down}% probability over next 5 trading days (model accuracy: {accuracy:.1f}%)",
            "icon": "🤖"
        })
    signals["ML Prediction"] = "bullish" if pred_val == 1 else "bearish"

    # ── 2. RSI Analysis ───────────────────────────────────────────────────────
    rsi = indicators.get("rsi", 50)
    if rsi < 30:
        bull_points += 2
        reasons.append({"category": "RSI", "signal": "bullish",
            "text": f"RSI at {rsi} — deeply oversold territory. Stock is historically cheap relative to recent price action. Often a strong buying opportunity.",
            "icon": "📊"})
    elif rsi < 45:
        bull_points += 1
        reasons.append({"category": "RSI", "signal": "bullish",
            "text": f"RSI at {rsi} — approaching oversold zone. Downward pressure may be easing soon.", "icon": "📊"})
    elif rsi > 70:
        bear_points += 2
        reasons.append({"category": "RSI", "signal": "bearish",
            "text": f"RSI at {rsi} — overbought territory. Stock may be due for a pullback or price correction in near term.",
            "icon": "📊"})
    elif rsi > 60:
        bear_points += 1
        reasons.append({"category": "RSI", "signal": "bearish",
            "text": f"RSI at {rsi} — approaching overbought zone. Upside may be limited.", "icon": "📊"})
    else:
        reasons.append({"category": "RSI", "signal": "neutral",
            "text": f"RSI at {rsi} — neutral zone (30-70). No extreme buying or selling pressure detected.",
            "icon": "📊"})
    signals["RSI"] = "bullish" if rsi < 45 else "bearish" if rsi > 65 else "neutral"

    # ── 3. MACD Signal ────────────────────────────────────────────────────────
    macd      = indicators.get("macd", 0)
    macd_sig  = indicators.get("macd_signal", 0)
    macd_hist = indicators.get("macd_hist", 0)

    if macd > macd_sig and macd_hist > 0:
        bull_points += 2
        reasons.append({"category": "MACD", "signal": "bullish",
            "text": f"MACD ({macd:.3f}) is above signal line — bullish crossover confirmed. Momentum is building upward.",
            "icon": "📈"})
    elif macd < macd_sig and macd_hist < 0:
        bear_points += 2
        reasons.append({"category": "MACD", "signal": "bearish",
            "text": f"MACD ({macd:.3f}) is below signal line — bearish crossover. Downward momentum is dominant.",
            "icon": "📉"})
    else:
        reasons.append({"category": "MACD", "signal": "neutral",
            "text": f"MACD ({macd:.3f}) near signal line — no clear momentum direction. Wait for clearer divergence.",
            "icon": "➖"})
    signals["MACD"] = "bullish" if macd > macd_sig else "bearish" if macd < macd_sig else "neutral"

    # ── 4. Moving Average Trend ───────────────────────────────────────────────
    price   = live_price.get("current_price", 0)
    ma21    = indicators.get("ma21", 0)
    ma50    = indicators.get("ma50", 0)
    pct_ma21 = indicators.get("price_vs_ma21", 0)
    pct_ma50 = indicators.get("price_vs_ma50", 0)

    if price > ma21 > ma50:
        bull_points += 2
        reasons.append({"category": "Moving Averages", "signal": "bullish",
            "text": f"Price (₹{price:,.2f}) > MA21 (₹{ma21:,.0f}) > MA50 (₹{ma50:,.0f}) — classic bullish alignment. Uptrend is intact.",
            "icon": "📈"})
    elif price < ma21 < ma50:
        bear_points += 2
        reasons.append({"category": "Moving Averages", "signal": "bearish",
            "text": f"Price (₹{price:,.2f}) < MA21 (₹{ma21:,.0f}) < MA50 (₹{ma50:,.0f}) — bearish alignment. Downtrend is intact.",
            "icon": "📉"})
    elif price > ma21:
        bull_points += 1
        reasons.append({"category": "Moving Averages", "signal": "bullish",
            "text": f"Price is {pct_ma21:+.1f}% above 21-day moving average — short-term uptrend holds.",
            "icon": "📈"})
    elif price < ma21:
        bear_points += 1
        reasons.append({"category": "Moving Averages", "signal": "bearish",
            "text": f"Price is {pct_ma21:+.1f}% below 21-day moving average — short-term downtrend persists.",
            "icon": "📉"})
    signals["Moving Averages"] = "bullish" if price > ma21 else "bearish"

    # ── 5. News Sentiment ─────────────────────────────────────────────────────
    news_score = news_sentiment.get("overall_score", 0)
    pos_c = news_sentiment.get("positive_count", 0)
    neg_c = news_sentiment.get("negative_count", 0)
    total = news_sentiment.get("total_articles", 1)

    if news_score >= 0.15:
        bull_points += 2
        reasons.append({"category": "News Sentiment", "signal": "bullish",
            "text": f"Strong positive news sentiment (score: {news_score:+.3f}). {pos_c} of {total} headlines are positive — market narrative favours the stock.",
            "icon": "📰"})
    elif news_score >= 0.05:
        bull_points += 1
        reasons.append({"category": "News Sentiment", "signal": "bullish",
            "text": f"Mildly positive news sentiment (score: {news_score:+.3f}). More positive than negative coverage in recent news.",
            "icon": "📰"})
    elif news_score <= -0.15:
        bear_points += 2
        reasons.append({"category": "News Sentiment", "signal": "bearish",
            "text": f"Strong negative news sentiment (score: {news_score:+.3f}). {neg_c} of {total} headlines are negative — market narrative unfavourable.",
            "icon": "📰"})
    elif news_score <= -0.05:
        bear_points += 1
        reasons.append({"category": "News Sentiment", "signal": "bearish",
            "text": f"Mild negative news sentiment (score: {news_score:+.3f}). Caution advised — news flow is slightly unfavourable.",
            "icon": "📰"})
    else:
        reasons.append({"category": "News Sentiment", "signal": "neutral",
            "text": f"Neutral news sentiment (score: {news_score:+.3f}). Mixed or low-impact news — no strong directional signal from media.",
            "icon": "📰"})
    signals["News Sentiment"] = "bullish" if news_score >= 0.05 else "bearish" if news_score <= -0.05 else "neutral"

    # ── 6. Fundamental Analysis ───────────────────────────────────────────────
    pe      = fundamentals.get("pe_ratio", 0)
    pb      = fundamentals.get("pb_ratio", 0)
    roe     = fundamentals.get("roe", 0)
    de      = fundamentals.get("debt_equity", 0)
    rev_g   = fundamentals.get("revenue_growth", 0)
    earn_g  = fundamentals.get("earnings_growth", 0)
    div_y   = fundamentals.get("dividend_yield", 0)
    margin  = fundamentals.get("profit_margin", 0)

    fund_bull = 0
    fund_bear = 0
    fund_texts = []

    if 0 < pe < 20:
        fund_bull += 1
        fund_texts.append(f"PE ratio of {pe:.1f}x is attractively valued (below 20x)")
    elif pe > 40:
        fund_bear += 1
        fund_texts.append(f"PE ratio of {pe:.1f}x is expensive — high growth expectation priced in")
    elif pe > 0:
        fund_texts.append(f"PE ratio of {pe:.1f}x is fair/moderate")

    if roe > 20:
        fund_bull += 1
        fund_texts.append(f"Excellent ROE of {roe:.1f}% — strong capital efficiency")
    elif roe > 12:
        fund_texts.append(f"Decent ROE of {roe:.1f}%")
    elif roe > 0:
        fund_bear += 1
        fund_texts.append(f"ROE of {roe:.1f}% is below par — management not generating strong returns")

    if de > 2.0:
        fund_bear += 1
        fund_texts.append(f"High Debt/Equity of {de:.1f}x — elevated financial risk")
    elif de < 0.5 and de >= 0:
        fund_bull += 1
        fund_texts.append(f"Low Debt/Equity of {de:.1f}x — strong balance sheet")

    if rev_g > 15:
        fund_bull += 1
        fund_texts.append(f"Strong revenue growth of {rev_g:.1f}% YoY")
    elif rev_g < -5:
        fund_bear += 1
        fund_texts.append(f"Revenue declining {rev_g:.1f}% YoY — business facing headwinds")

    if div_y > 2:
        fund_bull += 1
        fund_texts.append(f"Dividend yield of {div_y:.2f}% provides income support")

    if margin > 20:
        fund_bull += 1
        fund_texts.append(f"High profit margin of {margin:.1f}% — efficient operations")
    elif margin < 5 and margin >= 0:
        fund_bear += 1
        fund_texts.append(f"Thin profit margin of {margin:.1f}% — vulnerable to cost pressures")

    fund_signal = "bullish" if fund_bull > fund_bear else "bearish" if fund_bear > fund_bull else "neutral"
    bull_points += fund_bull
    bear_points += fund_bear

    if fund_texts:
        reasons.append({"category": "Fundamentals", "signal": fund_signal,
            "text": " | ".join(fund_texts), "icon": "🏢"})
    signals["Fundamentals"] = fund_signal

    # ── 7. Historical Returns ─────────────────────────────────────────────────
    r1m = period_returns.get("return_1m") or 0
    r3m = period_returns.get("return_3m") or 0
    r6m = period_returns.get("return_6m") or 0
    r1y = period_returns.get("return_1y") or 0

    if r1m > 5 and r3m > 10:
        bull_points += 1
        reasons.append({"category": "Price History", "signal": "bullish",
            "text": f"Strong recent performance: +{r1m:.1f}% (1M), +{r3m:.1f}% (3M), {r6m:+.1f}% (6M), {r1y:+.1f}% (1Y). Uptrend is sustained.",
            "icon": "📅"})
    elif r1m < -5 and r3m < -10:
        bear_points += 1
        reasons.append({"category": "Price History", "signal": "bearish",
            "text": f"Sustained weakness: {r1m:.1f}% (1M), {r3m:.1f}% (3M), {r6m:+.1f}% (6M), {r1y:+.1f}% (1Y). Downtrend in multiple timeframes.",
            "icon": "📅"})
    else:
        reasons.append({"category": "Price History", "signal": "neutral",
            "text": f"Mixed returns: {r1m:+.1f}% (1M), {r3m:+.1f}% (3M), {r6m:+.1f}% (6M), {r1y:+.1f}% (1Y). No clear sustained direction.",
            "icon": "📅"})
    signals["Price Trend"] = "bullish" if r1m > 3 else "bearish" if r1m < -3 else "neutral"

    # ── 8. 52-Week Position ───────────────────────────────────────────────────
    w52h = live_price.get("fifty_two_high", 0)
    w52l = live_price.get("fifty_two_low", 0)
    if w52h and w52l and price:
        range_52 = w52h - w52l
        pos_pct  = ((price - w52l) / range_52 * 100) if range_52 else 50
        if pos_pct < 30:
            bull_points += 1
            reasons.append({"category": "52-Week Range", "signal": "bullish",
                "text": f"Price is near 52-week LOW (₹{w52l:,.0f}–₹{w52h:,.0f}). At {pos_pct:.0f}% of range — potential value zone.",
                "icon": "📍"})
        elif pos_pct > 80:
            bear_points += 1
            reasons.append({"category": "52-Week Range", "signal": "bearish",
                "text": f"Price near 52-week HIGH (₹{w52l:,.0f}–₹{w52h:,.0f}). At {pos_pct:.0f}% of range — limited upside, high resistance.",
                "icon": "📍"})
        signals["52W Range"] = "bullish" if pos_pct < 30 else "bearish" if pos_pct > 80 else "neutral"

    # ── 9. Volume Analysis ────────────────────────────────────────────────────
    vol_ratio = indicators.get("volume_ratio", 1)
    vol_surge = period_returns.get("vol_surge", 1)
    if vol_ratio > 1.5:
        if pred_val == 1:
            bull_points += 1
            reasons.append({"category": "Volume", "signal": "bullish",
                "text": f"Volume is {vol_ratio:.1f}x above average — high conviction buying. Strong institutional or retail interest.",
                "icon": "📊"})
        else:
            bear_points += 1
            reasons.append({"category": "Volume", "signal": "bearish",
                "text": f"Volume is {vol_ratio:.1f}x above average during downward move — high conviction selling pressure.",
                "icon": "📊"})
    signals["Volume"] = "bullish" if vol_ratio > 1.3 and pred_val == 1 else "neutral"

    # ── Final Score & Verdict ──────────────────────────────────────────────────
    total_points = bull_points + bear_points
    bull_pct = (bull_points / total_points * 100) if total_points else 50
    bear_pct = (bear_points / total_points * 100) if total_points else 50

    if bull_pct >= 65:
        trend  = "BULLISH"
        emoji  = "🟢"
        action = "BUY / ACCUMULATE"
        action_color = "green"
        description = _build_buy_desc(bull_pct, indicators, fundamentals, period_returns, news_sentiment)
    elif bear_pct >= 65:
        trend  = "BEARISH"
        emoji  = "🔴"
        action = "AVOID / REDUCE"
        action_color = "red"
        description = _build_avoid_desc(bear_pct, indicators, fundamentals, period_returns, news_sentiment)
    elif bull_pct >= 55:
        trend  = "MILDLY BULLISH"
        emoji  = "🟡"
        action = "WAIT & WATCH"
        action_color = "yellow"
        description = _build_wait_desc(bull_pct, indicators, fundamentals, period_returns, news_sentiment)
    elif bear_pct >= 55:
        trend  = "MILDLY BEARISH"
        emoji  = "🟡"
        action = "WAIT FOR DIP"
        action_color = "yellow"
        description = _build_wait_desc(bull_pct, indicators, fundamentals, period_returns, news_sentiment)
    else:
        trend  = "NEUTRAL"
        emoji  = "⚪"
        action = "WAIT & WATCH"
        action_color = "yellow"
        description = _build_neutral_desc(indicators, period_returns)

    return {
        "trend":        trend,
        "emoji":        emoji,
        "action":       action,
        "action_color": action_color,
        "bull_score":   round(bull_pct, 1),
        "bear_score":   round(bear_pct, 1),
        "bull_points":  bull_points,
        "bear_points":  bear_points,
        "description":  description,
        "reasons":      reasons,
        "signals":      signals,
        "forecast_days": 5,
    }


def _build_buy_desc(score, ind, fund, ret, news):
    rsi  = ind.get("rsi", 50)
    r1m  = ret.get("return_1m") or 0
    ns   = news.get("overall_score", 0)
    pe   = fund.get("pe_ratio", 0)
    
    pe_str   = f" at a reasonable PE of {pe:.1f}x" if 0 < pe < 30 else ""
    news_str = "supported by positive news flow" if ns > 0.05 else "despite mixed news"
    
    return (
        f"Multiple signals align bullishly. The stock shows strong upward momentum "
        f"with RSI at {rsi:.0f} (healthy range){pe_str}, {news_str}. "
        f"The Random Forest model, trained on {3} years of price data with 27 technical features, "
        f"predicts continued upward movement for the next 5 trading days. "
        f"1-month return of {r1m:+.1f}% confirms the current trend is intact. "
        f"Consider entering at current levels or on minor dips. "
        f"Use a stop-loss of 3–5% below entry. This is an academic analysis — "
        f"always consult a SEBI-registered financial advisor before investing."
    )


def _build_avoid_desc(score, ind, fund, ret, news):
    rsi  = ind.get("rsi", 50)
    r1m  = ret.get("return_1m") or 0
    ns   = news.get("overall_score", 0)
    de   = fund.get("debt_equity", 0)
    
    debt_str = f" High debt/equity ({de:.1f}x) adds financial risk." if de > 1.5 else ""
    news_str = " Negative news sentiment adds to downward pressure." if ns < -0.05 else ""
    
    return (
        f"Most signals point bearish. RSI at {rsi:.0f} and 1-month return of {r1m:+.1f}% "
        f"reflect persistent selling pressure.{debt_str}{news_str} "
        f"The ML model confirms bearish momentum for the next 5 trading days. "
        f"Avoid fresh positions at current levels. Existing investors may consider "
        f"reducing exposure or setting tight stop-losses. "
        f"Wait for RSI to fall below 35 and sentiment to turn positive before re-entry. "
        f"This is an academic analysis — always verify with a SEBI-registered advisor."
    )


def _build_wait_desc(score, ind, fund, ret, news):
    rsi = ind.get("rsi", 50)
    r3m = ret.get("return_3m") or 0
    
    return (
        f"Signals are mixed — neither strongly bullish nor bearish. RSI at {rsi:.0f} "
        f"and 3-month return of {r3m:+.1f}% indicate consolidation phase. "
        f"The stock is building a base, but a clear breakout or breakdown signal is not yet confirmed. "
        f"Strategy: Wait for price to break above the 21-day MA with high volume for a bullish entry, "
        f"or fall below 52-week low support for a bearish exit. "
        f"Set price alerts rather than entering blindly. "
        f"This is an academic analysis — consult a SEBI-registered financial advisor."
    )


def _build_neutral_desc(ind, ret):
    rsi = ind.get("rsi", 50)
    r1y = ret.get("return_1y") or 0
    
    return (
        f"The stock is in a neutral, sideways zone. RSI at {rsi:.0f} and 1-year return of {r1y:+.1f}% "
        f"suggest range-bound trading. Technical and fundamental signals are evenly balanced. "
        f"No clear directional edge exists at this moment. "
        f"Patience is advised — wait for a catalyst (earnings result, policy announcement, "
        f"sector rotation) before taking a position. "
        f"Monitor weekly for clearer signal formation."
    )
