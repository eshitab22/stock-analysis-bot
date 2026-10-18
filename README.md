# 📈 Stock Analysis Bot

**🔗 Live demo: https://stock-analysis-bot-36xwnvxzgbwsjwfnxoswzg.streamlit.app/**

An interactive Streamlit app that, given a stock symbol, shows:

- **Multi-timeframe charts:** 1m, 3m, 5m, 15m, 1h, daily, weekly, monthly (6 months to all history)
- **Fundamental analysis:** P/E, P/B, ROE, debt/equity, growth, margins, with a transparent rule-based score
- **Technical analysis:** SMA 20/50/200, RSI, MACD, plus an intraday VWAP/EMA bias
- **ML next-day direction model:** Random Forest validated with **walk-forward splits**, compared against a naive baseline, with a cost-adjusted backtest
- **Verdict:** next day, intraday, swing, and long-term view

## Why the ML section is built this way
Next-day stock direction is close to a coin flip. Instead of claiming a high accuracy, the app
reports out-of-sample accuracy **versus a majority-class baseline**, and warns when there is no real edge.
There is no look-ahead leakage: features use only past data and splits respect time order.

## Run locally
```bash
git clone https://github.com/eshitab22/stock-analysis-bot.git
cd stock-analysis-bot
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Data comes from Yahoo Finance via `yfinance`. Intraday history is limited (1m: 7 days, 5m/15m: 60 days).
- Indian stocks: choose NSE/BSE in the sidebar (e.g. `RELIANCE`, `TCS`).
- The free hosted app may take about 30 seconds to wake up, and Yahoo may occasionally rate-limit it. If data looks empty, wait a minute and click Analyse again.
- Educational project, not financial advice.
