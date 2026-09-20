# Algory QI Homework 1

# Pulls 1yr of closes daily for SCHW and SPY

import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "SCHW"  # my brokerage account is at Schwab
BENCH = "SPY"

schw = yf.Ticker(TICKER).history(period="1y")["Close"].dropna()
spy = yf.Ticker(BENCH).history(period="1y")["Close"].dropna()

# Both tickers on same trading days
prices = pd.DataFrame({TICKER: schw, BENCH: spy}).dropna()
prices.index = prices.index.tz_localize(None)  # drop timezone so dates print clean

print(f"Tickers: {TICKER} and {BENCH}")
print(f"Trading days: {len(prices)}")
print(f"First date: {prices.index[0].date()}")
print(f"Last date: {prices.index[-1].date()}")
print()

for t in prices.columns:
    p = prices[t]
    ret = p.iloc[-1] / p.iloc[0] - 1
    vol = p.pct_change().dropna().std() * np.sqrt(252)
    print(f"{t} last close {p.iloc[-1]:.2f} return {ret:+.1%} ann vol {vol:.1%}")

# Biggest single-day move for schwab
daily = prices[TICKER].pct_change().dropna()
day = daily.abs().idxmax()
print(f"\nBiggest single-day move for {TICKER}: {daily.loc[day]:+.2%} on {day.date()}")

# Saves to excel
prices.index.name = "Date"
prices.to_csv("prices.csv")
