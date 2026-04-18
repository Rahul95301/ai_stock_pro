# =============================================================================
# src/stock_search.py — Smart Stock Search with Autocomplete
# FIXED v2: Multi-word queries like "hdfc bank", "tata motors" now work
# =============================================================================

import sys
import os

NSE_STOCKS = [
    ("RELIANCE.NS",    "Reliance Industries Ltd",           ["reliance", "ril", "reli"]),
    ("TCS.NS",         "Tata Consultancy Services",         ["tcs", "tata consultancy"]),
    ("HDFCBANK.NS",    "HDFC Bank Ltd",                     ["hdfc bank", "hdfcbank", "hdfc"]),
    ("INFY.NS",        "Infosys Ltd",                       ["infosys", "infy"]),
    ("ICICIBANK.NS",   "ICICI Bank Ltd",                    ["icici bank", "icicibank", "icici"]),
    ("HINDUNILVR.NS",  "Hindustan Unilever Ltd",            ["hul", "hindustan unilever", "hindustan"]),
    ("ITC.NS",         "ITC Ltd",                           ["itc"]),
    ("SBIN.NS",        "State Bank of India",               ["sbi", "state bank", "sbin"]),
    ("BHARTIARTL.NS",  "Bharti Airtel Ltd",                 ["airtel", "bharti"]),
    ("KOTAKBANK.NS",   "Kotak Mahindra Bank",               ["kotak", "kotak bank", "kotak mahindra"]),
    ("WIPRO.NS",       "Wipro Ltd",                         ["wipro"]),
    ("LT.NS",          "Larsen & Toubro Ltd",               ["l&t", "larsen", "lt", "l and t"]),
    ("BAJFINANCE.NS",  "Bajaj Finance Ltd",                 ["bajaj finance", "bajfinance"]),
    ("HCLTECH.NS",     "HCL Technologies Ltd",              ["hcl", "hcltech", "hcl tech"]),
    ("ASIANPAINT.NS",  "Asian Paints Ltd",                  ["asian paints", "asian paint"]),
    ("MARUTI.NS",      "Maruti Suzuki India Ltd",           ["maruti", "suzuki", "maruti suzuki"]),
    ("SUNPHARMA.NS",   "Sun Pharmaceutical Industries",     ["sun pharma", "sunpharma", "sun pharmaceutical"]),
    ("TATAMOTORS.NS",  "Tata Motors Ltd",                   ["tata motors", "tatamotors", "tata motor"]),
    ("ULTRACEMCO.NS",  "UltraTech Cement Ltd",              ["ultratech", "ultratech cement", "ultra tech"]),
    ("TITAN.NS",       "Titan Company Ltd",                 ["titan"]),
    ("NESTLEIND.NS",   "Nestle India Ltd",                  ["nestle", "maggi"]),
    ("POWERGRID.NS",   "Power Grid Corp of India",          ["power grid", "powergrid"]),
    ("NTPC.NS",        "NTPC Ltd",                          ["ntpc"]),
    ("ONGC.NS",        "Oil & Natural Gas Corporation",     ["ongc", "oil gas"]),
    ("TECHM.NS",       "Tech Mahindra Ltd",                 ["tech mahindra", "techm", "tech m"]),
    ("BAJAJFINSV.NS",  "Bajaj Finserv Ltd",                 ["bajaj finserv", "bajajfinsv", "bajaj fin"]),
    ("DIVISLAB.NS",    "Divi's Laboratories Ltd",           ["divis", "divis lab", "divi"]),
    ("DRREDDY.NS",     "Dr. Reddy's Laboratories",          ["dr reddy", "drreddy", "dr reddys"]),
    ("CIPLA.NS",       "Cipla Ltd",                         ["cipla"]),
    ("EICHERMOT.NS",   "Eicher Motors Ltd",                 ["eicher", "royal enfield"]),
    ("GRASIM.NS",      "Grasim Industries Ltd",             ["grasim"]),
    ("ADANIENT.NS",    "Adani Enterprises Ltd",             ["adani", "adani enterprises"]),
    ("ADANIPORTS.NS",  "Adani Ports & SEZ Ltd",             ["adani ports", "adaniports"]),
    ("COALINDIA.NS",   "Coal India Ltd",                    ["coal india", "coalindia"]),
    ("BPCL.NS",        "Bharat Petroleum Corp",             ["bpcl", "bharat petroleum"]),
    ("HINDALCO.NS",    "Hindalco Industries Ltd",           ["hindalco"]),
    ("JSWSTEEL.NS",    "JSW Steel Ltd",                     ["jsw", "jsw steel"]),
    ("TATASTEEL.NS",   "Tata Steel Ltd",                    ["tata steel", "tatasteel"]),
    ("INDUSINDBK.NS",  "IndusInd Bank Ltd",                 ["indusind", "indusind bank"]),
    ("APOLLOHOSP.NS",  "Apollo Hospitals Enterprise",       ["apollo", "apollo hospital", "apollo hospitals"]),
    ("SBILIFE.NS",     "SBI Life Insurance",                ["sbi life", "sbilife"]),
    ("HDFCLIFE.NS",    "HDFC Life Insurance",               ["hdfc life", "hdfclife"]),
    ("BAJAJ-AUTO.NS",  "Bajaj Auto Ltd",                    ["bajaj auto", "bajaj"]),
    ("M&M.NS",         "Mahindra & Mahindra Ltd",           ["mahindra", "m&m", "mahindra mahindra"]),
    ("HEROMOTOCO.NS",  "Hero MotoCorp Ltd",                 ["hero", "hero motocorp", "hero moto"]),
    ("BRITANNIA.NS",   "Britannia Industries Ltd",          ["britannia"]),
    ("TATACONSUM.NS",  "Tata Consumer Products",            ["tata consumer", "tataconsum"]),
    ("ZOMATO.NS",      "Zomato Ltd",                        ["zomato"]),
    ("PAYTM.NS",       "One97 Communications (Paytm)",      ["paytm"]),
    ("NYKAA.NS",       "FSN E-Commerce Ventures (Nykaa)",   ["nykaa"]),
    ("DMART.NS",       "Avenue Supermarts (DMart)",         ["dmart", "d-mart", "avenue", "dmart"]),
    ("PVR.NS",         "PVR Inox Ltd",                      ["pvr", "pvr inox", "inox"]),
    ("IRCTC.NS",       "Indian Railway Catering & Tourism", ["irctc", "railway", "indian railway"]),
    ("POLYCAB.NS",     "Polycab India Ltd",                 ["polycab"]),
    ("MPHASIS.NS",     "Mphasis Ltd",                       ["mphasis"]),
    ("PERSISTENT.NS",  "Persistent Systems Ltd",            ["persistent"]),
    ("COFORGE.NS",     "Coforge Ltd",                       ["coforge"]),
    ("LTIM.NS",        "LTIMindtree Ltd",                   ["ltimindtree", "ltim", "lti mindtree"]),
    ("PAGEIND.NS",     "Page Industries Ltd",               ["page industries", "jockey"]),
    ("TATAPOWER.NS",   "Tata Power Company",                ["tata power", "tatapower"]),
    ("TRENT.NS",       "Trent Ltd",                         ["trent", "westside", "zudio"]),
    ("HAVELLS.NS",     "Havells India Ltd",                 ["havells"]),
    ("VOLTAS.NS",      "Voltas Ltd",                        ["voltas"]),
    ("GODREJCP.NS",    "Godrej Consumer Products",          ["godrej", "godrejcp", "godrej consumer"]),
    ("AMBUJACEM.NS",   "Ambuja Cements Ltd",                ["ambuja", "ambuja cement"]),
    ("ACC.NS",         "ACC Ltd",                           ["acc cement", "acc"]),
    ("SHREECEM.NS",    "Shree Cement Ltd",                  ["shree cement", "shree"]),
    ("BANKBARODA.NS",  "Bank of Baroda",                    ["bob", "bank of baroda", "bankbaroda", "baroda"]),
    ("PNB.NS",         "Punjab National Bank",              ["pnb", "punjab national", "punjab bank"]),
    ("CANBK.NS",       "Canara Bank",                       ["canara", "canara bank"]),
    ("FEDERALBNK.NS",  "Federal Bank Ltd",                  ["federal bank", "federal"]),
    ("IDFCFIRSTB.NS",  "IDFC First Bank",                   ["idfc", "idfc first", "idfc bank"]),
    ("BANDHANBNK.NS",  "Bandhan Bank Ltd",                  ["bandhan bank", "bandhan"]),
    ("MANAPPURAM.NS",  "Manappuram Finance",                ["manappuram"]),
    ("CHOLAFIN.NS",    "Cholamandalam Investment",          ["chola", "cholamandalam"]),
    ("MOTHERSON.NS",   "Samvardhana Motherson",             ["motherson", "samvardhana"]),
    ("BALKRISIND.NS",  "Balkrishna Industries",             ["bkt", "balkrishna"]),
    ("ASTRAL.NS",      "Astral Ltd",                        ["astral"]),
    ("RELAXO.NS",      "Relaxo Footwears Ltd",              ["relaxo", "relaxo shoes"]),
    ("BERGEPAINT.NS",  "Berger Paints India",               ["berger", "berger paint"]),
    ("CROMPTON.NS",    "Crompton Greaves Consumer",         ["crompton"]),
    ("LICHSGFIN.NS",   "LIC Housing Finance",               ["lic housing", "lichsgfin", "lic"]),
    ("RECLTD.NS",      "REC Ltd",                           ["rec", "rural electrification"]),
    ("PFC.NS",         "Power Finance Corporation",         ["pfc", "power finance"]),
    ("NHPC.NS",        "NHPC Ltd",                          ["nhpc"]),
    ("IRFC.NS",        "Indian Railway Finance Corp",       ["irfc"]),
    ("HAL.NS",         "Hindustan Aeronautics Ltd",         ["hal", "hindustan aeronautics"]),
    ("BEL.NS",         "Bharat Electronics Ltd",            ["bel", "bharat electronics"]),
    ("BHEL.NS",        "Bharat Heavy Electricals Ltd",      ["bhel", "bharat heavy"]),
    ("SAIL.NS",        "Steel Authority of India",          ["sail", "steel authority"]),
    ("NMDC.NS",        "NMDC Ltd",                          ["nmdc"]),
    ("GAIL.NS",        "GAIL India Ltd",                    ["gail"]),
    ("IOC.NS",         "Indian Oil Corporation",            ["ioc", "indian oil"]),
    ("HINDPETRO.NS",   "Hindustan Petroleum Corp",          ["hpcl", "hindustan petroleum", "hindpetro"]),
    ("INDIAMART.NS",   "IndiaMart InterMesh",               ["indiamart"]),
    ("MARICO.NS",      "Marico Ltd",                        ["marico", "parachute", "saffola"]),
    ("DABUR.NS",       "Dabur India Ltd",                   ["dabur"]),
    ("COLPAL.NS",      "Colgate-Palmolive India",           ["colgate", "colpal"]),
    ("EMAMILTD.NS",    "Emami Ltd",                         ["emami"]),
]


def _score_match(query: str, ticker: str, name: str, aliases: list) -> int:
    """
    Multi-strategy scoring — handles:
      "hdfc bank"   → HDFCBANK.NS  (multi-word)
      "tata motors" → TATAMOTORS.NS
      "hdfcb"       → HDFCBANK.NS  (partial ticker)
      "hdfc"        → HDFCBANK + HDFCLIFE (both)
      "sbi"         → SBIN.NS + SBILIFE.NS
    """
    q         = query.lower().strip()
    q_nospace = q.replace(" ", "")        # "hdfc bank" → "hdfcbank"
    q_words   = [w for w in q.split() if w]  # ["hdfc","bank"]
    t_clean   = ticker.lower().replace(".ns", "")
    n_lower   = name.lower()
    score     = 0

    # ── Exact matches ────────────────────────────────────────────────────────
    if q == t_clean:                              return 100
    if q_nospace == t_clean:                      return 98
    if q in [a.lower() for a in aliases]:         return 95
    if q == n_lower:                              return 90

    # ── Ticker starts-with (handles partial tickers like "hdfcb") ────────────
    if t_clean.startswith(q):                     score = max(score, 85)
    if q_nospace and t_clean.startswith(q_nospace): score = max(score, 85)

    # ── ALL words present in company name ("hdfc bank" in "hdfc bank ltd") ───
    if q_words and all(w in n_lower for w in q_words):
        score = max(score, 82)

    # ── ALL words present in any alias ───────────────────────────────────────
    for alias in aliases:
        a = alias.lower()
        if q_words and all(w in a for w in q_words):
            score = max(score, 80)
        if a.startswith(q):
            score = max(score, 75)
        if q_nospace and a.replace(" ", "").startswith(q_nospace):
            score = max(score, 74)

    # ── Company name starts-with ──────────────────────────────────────────────
    if n_lower.startswith(q):                     score = max(score, 75)

    # ── No-space version in ticker (hdfcbank in hdfcbank.ns) ─────────────────
    if q_nospace and q_nospace in t_clean:        score = max(score, 65)

    # ── Query contained in company name ──────────────────────────────────────
    if q in n_lower:                              score = max(score, 50)

    # ── First word matches start of company name ──────────────────────────────
    if q_words and n_lower.startswith(q_words[0]):
        score = max(score, 45)

    # ── Any word (3+ chars) appears in company name ───────────────────────────
    for w in q_words:
        if len(w) >= 3 and w in n_lower:
            score = max(score, 38)
        if len(w) >= 3 and w in t_clean:
            score = max(score, 35)

    return score


def search_stocks(query: str, max_results: int = 8) -> list:
    """
    Searches NSE stock database and returns best matches sorted by score.
    """
    if not query or not query.strip():
        return []

    results = []
    for ticker, name, aliases in NSE_STOCKS:
        score = _score_match(query, ticker, name, aliases)
        if score > 0:
            results.append((ticker, name, score))

    results.sort(key=lambda x: (-x[2], x[1]))
    return results[:max_results]


def interactive_stock_search() -> str:
    """Terminal-based interactive stock search (used by main.py)."""
    _print_header()
    while True:
        query = input("\n  🔍 Search stock (name or ticker): ").strip()
        if not query:
            print("  ⚠  Please type something.")
            continue

        direct = query.upper()
        if "." not in direct:
            direct += ".NS"
        if any(t == direct for t, _, _ in NSE_STOCKS):
            print(f"\n  ✔  Selected: {direct}")
            return direct

        results = search_stocks(query)
        if not results:
            print(f"\n  ❌ No stocks found for '{query}'")
            retry = input("  Try again? (y/n): ").strip().lower()
            if retry != "y":
                ticker = query.upper()
                if "." not in ticker:
                    ticker += ".NS"
                return ticker
            continue

        print(f"\n  📋 Suggestions for '{query}':\n")
        print(f"  {'#':<4} {'Ticker':<18} {'Company Name'}")
        print(f"  {'─'*55}")
        for i, (ticker, name, _) in enumerate(results, 1):
            print(f"  {i}.   {ticker:<18} {name}")

        print(f"\n  Enter number to select, or press Enter to search again:")
        choice = input("  Your choice: ").strip()

        if choice.lower() == "q":
            sys.exit(0)
        if choice == "":
            continue
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                selected = results[idx][0]
                print(f"\n  ✅ Selected: {selected}  ({results[idx][1]})")
                return selected
            else:
                print(f"  ⚠  Enter 1 to {len(results)}.")
        else:
            query = choice


def _print_header():
    print("\n" + "=" * 60)
    print("  🤖 AI STOCK ANALYSIS TOOL")
    print("=" * 60)
    print()
    print("  📌 Search by name: reliance / tata / hdfc bank")
    print("  📌 Search by ticker: TCS / INFY / SBIN")
    print(f"\n  📊 Database: {len(NSE_STOCKS)} NSE stocks available")
