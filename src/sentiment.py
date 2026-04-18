# =============================================================================
# src/sentiment.py — News Sentiment Analysis Module
# Scores REAL RSS news headlines using VADER NLP
# Also generates historical sentiment series for ML training
# =============================================================================

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_headlines(news_items: list) -> list:
    """
    Takes list of {headline, source, relevant} dicts.
    Adds vader_score and sentiment_label to each.
    Returns enriched list sorted by abs(score) descending.
    """
    scored = []
    for item in news_items:
        headline = item.get("headline", "")
        scores   = _analyzer.polarity_scores(headline)
        compound = round(scores["compound"], 4)
        label    = ("Positive" if compound >= 0.05 else
                    "Negative" if compound <= -0.05 else "Neutral")
        scored.append({
            **item,
            "score": compound,
            "label": label,
            "pos":   round(scores["pos"], 3),
            "neg":   round(scores["neg"], 3),
            "neu":   round(scores["neu"], 3),
        })
    scored.sort(key=lambda x: abs(x["score"]), reverse=True)
    return scored


def compute_news_sentiment_summary(scored_news: list) -> dict:
    """
    Aggregate sentiment from all scored news.
    Returns overall_score, label, positive_count, negative_count, neutral_count.
    """
    if not scored_news:
        return {"overall_score": 0.0, "label": "Neutral",
                "positive_count": 0, "negative_count": 0, "neutral_count": 0,
                "company_score": 0.0, "market_score": 0.0}

    # Weight company-specific news more than general market news
    company_items = [n for n in scored_news if n.get("relevant")]
    market_items  = [n for n in scored_news if not n.get("relevant")]

    def avg_score(items):
        return np.mean([i["score"] for i in items]) if items else 0.0

    company_score = avg_score(company_items)
    market_score  = avg_score(market_items)

    # 70% weight company news, 30% market if we have both
    if company_items and market_items:
        overall = 0.7 * company_score + 0.3 * market_score
    elif company_items:
        overall = company_score
    else:
        overall = market_score

    overall = round(float(overall), 4)
    label   = "Positive" if overall >= 0.05 else "Negative" if overall <= -0.05 else "Neutral"

    pos_c = sum(1 for n in scored_news if n["label"] == "Positive")
    neg_c = sum(1 for n in scored_news if n["label"] == "Negative")
    neu_c = sum(1 for n in scored_news if n["label"] == "Neutral")

    return {
        "overall_score":   overall,
        "label":           label,
        "positive_count":  pos_c,
        "negative_count":  neg_c,
        "neutral_count":   neu_c,
        "company_score":   round(float(company_score), 4),
        "market_score":    round(float(market_score), 4),
        "total_articles":  len(scored_news),
    }


def build_sentiment_series(date_index: pd.DatetimeIndex,
                            ticker: str,
                            latest_score: float = 0.0) -> pd.DataFrame:
    """
    Generates a realistic daily sentiment time series for ML training.
    Anchors the most recent value to the actual fetched news score.
    Uses sine wave + Gaussian noise base, blended with VADER-scored templates.
    """
    print(f"  → Building sentiment series for {len(date_index)} trading days...")

    rng  = np.random.RandomState(42)
    n    = len(date_index)
    name = ticker.split(".")[0]

    # Base oscillation + noise
    base  = np.sin(np.linspace(0, 6 * np.pi, n)) * 0.25
    noise = rng.normal(0, 0.12, n)
    signal = np.clip(base + noise, -1, 1)

    # Blend with VADER on template headlines for realism
    TEMPLATES = [
        f"{name} Q{{q}} profit beats expectations",
        f"{name} signs strategic partnership deal",
        f"Regulatory concerns weigh on {name}",
        f"Broader market rally lifts {name}",
        f"{name} misses revenue target this quarter",
        f"FII buying in {name} signals confidence",
        f"{name} announces buyback program worth ₹2000 Cr",
        f"Rising costs pressure {name} margins",
        f"Analysts upgrade {name} with higher price target",
        f"Market selloff drags {name} lower",
    ]

    import random
    random.seed(42)

    records = []
    for i, date in enumerate(date_index):
        tmpl     = random.choice(TEMPLATES).format(q=random.randint(1, 4))
        v_score  = _analyzer.polarity_scores(tmpl)["compound"]
        blended  = round(0.5 * v_score + 0.5 * float(signal[i]), 4)

        # Anchor last 21 days to real news score (smooth transition)
        if i >= n - 21:
            weight = (i - (n - 21)) / 21  # 0 → 1 over last 21 days
            blended = round((1 - weight) * blended + weight * latest_score, 4)

        records.append({"Date": date, "headline": tmpl, "sentiment_compound": blended})

    df = pd.DataFrame(records).set_index("Date")
    print(f"  ✔  Sentiment series built (mean={df['sentiment_compound'].mean():.3f})")
    return df


def interpret_sentiment_score(score: float) -> str:
    if score >= 0.3:  return "Very Positive 🟢"
    if score >= 0.05: return "Positive 🟢"
    if score <= -0.3: return "Very Negative 🔴"
    if score <= -0.05:return "Negative 🔴"
    return "Neutral ⚪"
