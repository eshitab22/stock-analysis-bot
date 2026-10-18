# 📈 Stock Analysis Bot

An interactive Streamlit app that, given a stock symbol, shows:

- **Multi-timeframe charts:** 1m, 3m, 5m, 15m, 1h, daily, weekly, monthly (6 months to all history)
- **Fundamental analysis:** P/E, P/B, ROE, debt/equity, growth, margins, with a transparent rule-based score
- **Technical analysis:** SMA 20/50/200, RSI, MACD, plus an intraday VWAP/EMA bias
- **ML next-day direction model:** Random Forest validated with **walk-forward splits**, compared against a naive baseline, with a cost-adjusted backtest
- **Verdict:** next day, intraday, swing, and long-term view

## Why the ML section is built this way
Next-day stock direction is close to a coin flip. Instead of claiming a high accuracy, the app
reports out-of-sample accuracy **versus a majority-class baseline**, and warns when there is no real edge.
No look-ahead leakage: features use only past data and splits respect time order.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Data comes from Yahoo Finance via `yfinance`. Intraday history is limited (1m: 7 days, 5m/15m: 60 days).
- Indian stocks: choose NSE/BSE in the sidebar (e.g. `RELIANCE`, `TCS`).
- Educational project, not financial advice.
