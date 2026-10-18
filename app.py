"""
Stock Analysis Bot - multi-timeframe charts, fundamentals, technicals,
and an honestly-validated ML next-day direction model.

Run:  streamlit run app.py
Educational project. Not financial advice.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="Stock Analysis Bot", layout="wide")

# ----------------------------------------------------------------- data ----
# yfinance limits: 1m -> last 7 days, 5m/15m -> last 60 days, 60m -> ~730 days.
# 3m is not offered by Yahoo, so we download 1m and resample.
PRESETS = {
    "1 min  (last 5 days)": ("5d", "1m"),
    "3 min  (last 5 days)": ("5d", "3m"),
    "5 min  (last 1 month)": ("1mo", "5m"),
    "15 min (last 60 days)": ("60d", "15m"),
    "1 hour (last 6 months)": ("6mo", "60m"),
    "Daily  (last 6 months)": ("6mo", "1d"),
    "Daily  (last 1 year)": ("1y", "1d"),
    "Daily  (last 3 years)": ("3y", "1d"),
    "Daily  (last 5 years)": ("5y", "1d"),
    "Weekly (last 5 years)": ("5y", "1wk"),
    "Monthly (all history)": ("max", "1mo"),
}


@st.cache_data(ttl=300, show_spinner=False)
def load(ticker: str, period: str, interval: str) -> pd.DataFrame:
    fetch_interval = "1m" if interval == "3m" else interval
    df = yf.download(ticker, period=period, interval=fetch_interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if interval == "3m":
        df = df.resample("3min").agg({"Open": "first", "High": "max", "Low": "min",
                                      "Close": "last", "Volume": "sum"}).dropna()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


# ----------------------------------------------------------- indicators ----
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def macd_hist(close):
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return macd - macd.ewm(span=9, adjust=False).mean()


def add_indicators(df):
    d = df.copy()
    d["SMA20"] = d["Close"].rolling(20).mean()
    d["SMA50"] = d["Close"].rolling(50).mean()
    d["SMA200"] = d["Close"].rolling(200).mean()
    d["RSI"] = rsi(d["Close"])
    d["MACD_H"] = macd_hist(d["Close"])
    return d


# ---------------------------------------------------------------- charts ---
def candle_chart(df, title):
    d = add_indicators(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                                 close=d["Close"], name="Price"), row=1, col=1)
    for col, color in [("SMA20", "orange"), ("SMA50", "royalblue"), ("SMA200", "purple")]:
        if d[col].notna().any():
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col,
                                     line=dict(width=1, color=color)), row=1, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume", opacity=0.4), row=2, col=1)
    fig.update_layout(title=title, height=620, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def horizon_returns(daily):
    c = daily["Close"]
    rows = {}
    for label, n in [("1 day", 1), ("1 week", 5), ("1 month", 21), ("3 months", 63),
                     ("6 months", 126), ("1 year", 252), ("3 years", 756), ("5 years", 1260)]:
        if len(c) > n:
            rows[label] = f"{(c.iloc[-1] / c.iloc[-1 - n] - 1) * 100:+.2f}%"
        else:
            rows[label] = "n/a"
    return pd.DataFrame(rows, index=["Return"])


# ----------------------------------------------------------- fundamentals --
def fundamentals_table(info):
    def g(k):
        return info.get(k)

    def pct(v):
        return f"{v * 100:.2f}%" if isinstance(v, (int, float)) else "n/a"

    def num(v, d=2):
        return f"{v:,.{d}f}" if isinstance(v, (int, float)) else "n/a"

    rows = [
        ("Market cap", num(g("marketCap"), 0)),
        ("Trailing P/E", num(g("trailingPE"))),
        ("Forward P/E", num(g("forwardPE"))),
        ("Price / Book", num(g("priceToBook"))),
        ("Return on equity", pct(g("returnOnEquity"))),
        ("Profit margin", pct(g("profitMargins"))),
        ("Revenue growth (yoy)", pct(g("revenueGrowth"))),
        ("Earnings growth (yoy)", pct(g("earningsGrowth"))),
        ("Debt / Equity", num(g("debtToEquity"))),
        ("Current ratio", num(g("currentRatio"))),
        ("Dividend yield", num(g("dividendYield"))),
        ("Beta", num(g("beta"))),
        ("52-week high", num(g("fiftyTwoWeekHigh"))),
        ("52-week low", num(g("fiftyTwoWeekLow"))),
        ("Analyst target (mean)", num(g("targetMeanPrice"))),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")


def fundamental_score(info):
    """Simple transparent rule-based score. Returns (score, max, reasons)."""
    s, reasons = 0, []

    def check(cond, ok, bad):
        nonlocal s
        if cond is None:
            return
        if cond:
            s += 1
            reasons.append("✅ " + ok)
        else:
            reasons.append("⚠️ " + bad)

    pe, roe = info.get("trailingPE"), info.get("returnOnEquity")
    de, rg = info.get("debtToEquity"), info.get("revenueGrowth")
    pm, eg = info.get("profitMargins"), info.get("earningsGrowth")
    check(None if pe is None else 0 < pe < 30, "P/E is reasonable (<30)", "P/E is high or negative")
    check(None if roe is None else roe > 0.15, "ROE above 15%", "ROE below 15%")
    check(None if de is None else de < 100, "Debt/Equity under 100", "Debt/Equity is high")
    check(None if rg is None else rg > 0.05, "Revenue growing >5%", "Revenue growth weak")
    check(None if pm is None else pm > 0.10, "Profit margin above 10%", "Thin profit margin")
    check(None if eg is None else eg > 0, "Earnings growing", "Earnings shrinking")
    return s, len(reasons), reasons


# --------------------------------------------------------------- ML model --
def build_features(daily):
    d = daily.copy()
    r = d["Close"].pct_change()
    f = pd.DataFrame(index=d.index)
    for n in (1, 2, 3, 5):
        f[f"ret_lag{n}"] = r.shift(n - 1)
    f["ret_5d"] = d["Close"].pct_change(5)
    f["ret_10d"] = d["Close"].pct_change(10)
    f["rsi"] = rsi(d["Close"])
    f["macd_h"] = macd_hist(d["Close"]) / d["Close"]
    f["dist_sma20"] = d["Close"] / d["Close"].rolling(20).mean() - 1
    f["dist_sma50"] = d["Close"] / d["Close"].rolling(50).mean() - 1
    f["vol_20d"] = r.rolling(20).std()
    f["vol_ratio"] = d["Volume"] / d["Volume"].rolling(20).mean()
    f["hl_range"] = (d["High"] - d["Low"]) / d["Close"]
    target = (d["Close"].shift(-1) > d["Close"]).astype(float)
    target[d.index[-1]] = np.nan  # tomorrow is unknown
    return f, target, r


def make_model():
    # Shallow + heavy regularisation: financial data is noisy and overfits easily.
    return RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=30,
                                  n_jobs=-1, random_state=42)


def walk_forward(daily, cost_bps=5):
    f, y, r = build_features(daily)
    data = f.join(y.rename("y")).dropna()
    X, Y = data.drop(columns="y"), data["y"].astype(int)
    next_ret = r.shift(-1).reindex(data.index)

    proba = pd.Series(np.nan, index=data.index)
    for tr, te in TimeSeriesSplit(n_splits=5, gap=1).split(X):
        m = make_model().fit(X.iloc[tr], Y.iloc[tr])
        proba.iloc[te] = m.predict_proba(X.iloc[te])[:, 1]
    proba = proba.dropna()
    yt = Y.loc[proba.index]
    pred = (proba > 0.5).astype(int)

    acc = (pred == yt).mean()
    baseline = max(yt.mean(), 1 - yt.mean())  # always guess the majority class

    pos = pred.astype(float)
    cost = pos.diff().abs().fillna(0) * cost_bps / 10_000
    strat = pos * next_ret.loc[proba.index] - cost
    bh = next_ret.loc[proba.index]
    curves = pd.DataFrame({"Model strategy": (1 + strat).cumprod(),
                           "Buy & hold": (1 + bh).cumprod()})

    # Final model on all labelled data -> tomorrow's probability
    final = make_model().fit(X, Y)
    p_up = final.predict_proba(f.iloc[[-1]].ffill().fillna(0))[0, 1]
    imp = pd.Series(final.feature_importances_, index=X.columns).sort_values()
    return dict(acc=acc, baseline=baseline, n=len(yt), curves=curves, p_up=p_up, imp=imp)


# ----------------------------------------------------- rule-based verdicts --
def technical_view(daily):
    d = add_indicators(daily).dropna(subset=["SMA50"])
    last = d.iloc[-1]
    pts, notes = 0, []
    if last["Close"] > last["SMA50"]:
        pts += 1; notes.append("✅ Price above 50-day average")
    else:
        notes.append("⚠️ Price below 50-day average")
    if not np.isnan(last["SMA200"]):
        if last["SMA50"] > last["SMA200"]:
            pts += 1; notes.append("✅ 50-day above 200-day (long-term uptrend)")
        else:
            notes.append("⚠️ 50-day below 200-day (long-term downtrend)")
    if 40 <= last["RSI"] <= 70:
        pts += 1; notes.append(f"✅ RSI healthy ({last['RSI']:.0f})")
    elif last["RSI"] > 70:
        notes.append(f"⚠️ RSI overbought ({last['RSI']:.0f})")
    else:
        notes.append(f"⚠️ RSI weak/oversold ({last['RSI']:.0f})")
    if last["MACD_H"] > 0:
        pts += 1; notes.append("✅ MACD momentum positive")
    else:
        notes.append("⚠️ MACD momentum negative")
    return pts, 4, notes


def intraday_view(ticker):
    d = load(ticker, "5d", "5m")
    if d.empty:
        return None
    d = d.copy()
    d["day"] = d.index.date
    today = d[d["day"] == d["day"].iloc[-1]].copy()
    tp = (today["High"] + today["Low"] + today["Close"]) / 3
    today["VWAP"] = (tp * today["Volume"]).cumsum() / today["Volume"].cumsum()
    ema9 = today["Close"].ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = today["Close"].ewm(span=21, adjust=False).mean().iloc[-1]
    px, vwap = today["Close"].iloc[-1], today["VWAP"].iloc[-1]
    bias = "BULLISH" if (px > vwap and ema9 > ema21) else \
           "BEARISH" if (px < vwap and ema9 < ema21) else "NEUTRAL"
    return bias, px, vwap, ema9, ema21


def label(score, mx):
    r = score / mx if mx else 0
    return "Favourable" if r >= 0.67 else "Mixed" if r >= 0.34 else "Weak"


# ------------------------------------------------------------------- UI ----
st.title("📈 Stock Analysis Bot")
st.caption("Educational tool - not financial advice. Markets are noisy; treat every output as a hypothesis.")

with st.sidebar:
    st.header("Input")
    raw = st.text_input("Stock symbol", "RELIANCE").strip().upper()
    market = st.selectbox("Market", ["India NSE (.NS)", "India BSE (.BO)", "US / other (as typed)"])
    cost_bps = st.slider("Trading cost per position change (bps)", 0, 30, 5)
    go_btn = st.button("Analyse", type="primary")

if raw and (go_btn or "ran" in st.session_state):
    st.session_state["ran"] = True
    ticker = raw
    if "." not in raw:
        ticker = raw + (".NS" if market.startswith("India NSE") else ".BO" if "BSE" in market else "")

    daily = load(ticker, "10y", "1d")
    if daily.empty or len(daily) < 300:
        st.error(f"Not enough data for '{ticker}'. Check the symbol (e.g. TCS, INFY, AAPL).")
        st.stop()

    info = load_info(ticker)
    st.subheader(f"{info.get('longName', ticker)}  ({ticker})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Last close", f"{daily['Close'].iloc[-1]:,.2f}",
              f"{daily['Close'].pct_change().iloc[-1] * 100:+.2f}%")
    c2.metric("Sector", info.get("sector", "n/a"))
    c3.metric("Currency", info.get("currency", "n/a"))

    t_chart, t_fund, t_tech, t_ml, t_verdict = st.tabs(
        ["Charts", "Fundamentals", "Technicals & Intraday", "ML next-day prediction", "Verdict"])

    with t_chart:
        choice = st.selectbox("Timeframe", list(PRESETS))
        period, interval = PRESETS[choice]
        with st.spinner("Loading..."):
            df = load(ticker, period, interval)
        if df.empty:
            st.warning("No data returned (market may be closed or interval unsupported).")
        else:
            st.plotly_chart(candle_chart(df, choice), use_container_width=True)
        st.markdown("**Returns across horizons**")
        st.dataframe(horizon_returns(daily), use_container_width=True)

    with t_fund:
        fs, fm, freasons = fundamental_score(info)
        st.dataframe(fundamentals_table(info), use_container_width=True)
        if fm:
            st.markdown(f"**Fundamental score: {fs}/{fm}**")
            for x in freasons:
                st.write(x)
        else:
            st.info("Fundamentals unavailable for this symbol.")

    with t_tech:
        ts, tm, tnotes = technical_view(daily)
        st.markdown(f"**Technical score (daily): {ts}/{tm}**")
        for x in tnotes:
            st.write(x)
        iv = intraday_view(ticker)
        st.markdown("**Intraday (5-min, latest session)**")
        if iv:
            bias, px, vwap, e9, e21 = iv
            st.write(f"Bias: **{bias}**  |  Price {px:,.2f}  |  VWAP {vwap:,.2f}  |  "
                     f"EMA9 {e9:,.2f}  |  EMA21 {e21:,.2f}")
        else:
            st.info("No intraday data available right now.")

    with t_ml:
        with st.spinner("Training with walk-forward validation..."):
            res = walk_forward(daily, cost_bps)
        direction = "UP ⬆️" if res["p_up"] > 0.5 else "DOWN ⬇️"
        st.metric("Model's next-day call", direction, f"P(up) = {res['p_up']:.1%}")
        a, b, c = st.columns(3)
        a.metric("Out-of-sample accuracy", f"{res['acc']:.1%}")
        b.metric("Naive baseline", f"{res['baseline']:.1%}")
        c.metric("Edge vs baseline", f"{(res['acc'] - res['baseline']) * 100:+.1f} pts")
        st.caption(f"Based on {res['n']} out-of-sample days, expanding-window walk-forward "
                   "(never trained on the future).")
        st.line_chart(res["curves"])
        st.bar_chart(res["imp"])
        if res["acc"] - res["baseline"] < 0.02:
            st.warning("No meaningful edge over the naive baseline - the honest conclusion for "
                       "most liquid stocks. Do not trade this signal.")

    with t_verdict:
        st.markdown(f"**Next day:** {direction} (P(up) {res['p_up']:.0%}) - low confidence unless "
                    "the ML tab shows a real edge.")
        st.markdown(f"**Intraday:** {iv[0] if iv else 'n/a'} (VWAP + EMA rule)")
        st.markdown(f"**Swing / positional:** {label(ts, tm)} (technical score {ts}/{tm})")
        st.markdown(f"**Long-term investment:** {label(fs, fm) if fm else 'n/a'} "
                    f"(fundamental score {fs}/{fm}), supported by trend: {label(ts, tm)}")
        st.info("Rule-based scores are heuristics, not predictions. Always check news, "
                "valuation, and your own risk tolerance.")
else:
    st.info("Enter a symbol in the sidebar and press **Analyse**.")
