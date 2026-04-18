# =============================================================================
# config.py — Central Configuration
# AI Stock Advisor Pro | JECRC Sem VI Minor Project 2025-26
# =============================================================================

# ── Data Settings ─────────────────────────────────────────────────────────────
YEARS_OF_DATA     = 3          # Full 3-year historical data
TEST_SIZE         = 0.2
RANDOM_STATE      = 42
OUTPUT_DIR        = "outputs"

# ── Technical Indicator Windows ───────────────────────────────────────────────
MA_SHORT          = 7
MA_MEDIUM         = 21
MA_LONG           = 50
RSI_WINDOW        = 14
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
MOMENTUM_WINDOW   = 5
VOLATILITY_WINDOW = 14
BB_WINDOW         = 20
FORECAST_DAYS     = 5          # Predict next 5 trading days

# ── Random Forest Hyperparameters ─────────────────────────────────────────────
RF_N_ESTIMATORS     = 200
RF_MAX_DEPTH        = 7
RF_MIN_SAMPLES_LEAF = 5
RF_MIN_SAMPLES_SPLIT = 12

# ── Market Trend Thresholds ───────────────────────────────────────────────────
RSI_OVERBOUGHT    = 70
RSI_OVERSOLD      = 30
SENTIMENT_BULL    = 0.05
SENTIMENT_BEAR    = -0.05
MOMENTUM_BULL     = 1.0
MOMENTUM_BEAR     = -1.0

# ── News RSS Feeds (Free, No API Key Required) ────────────────────────────────
RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rss.cms",
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://feeds.feedburner.com/ndtvprofit-latest",
]
NEWS_FETCH_TIMEOUT = 8   # seconds
MAX_NEWS_ARTICLES  = 15
