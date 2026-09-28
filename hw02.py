# Algory QI Hw2

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def present_value(cash_flows, rate):
    total = 0
    for t in range(1, len(cash_flows) + 1):
        total = total + cash_flows[t - 1] / (1 + rate) ** t
    return total


def bond_price(face, coupon_rate, years, market_rate):
    coupon = face * coupon_rate
    flows = [coupon] * years
    flows[-1] = flows[-1] + face
    return present_value(flows, market_rate)


def annualised_return(prices):
    p = pd.Series(list(prices))
    years = (len(p) - 1) / 252
    return (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1


def annualised_volatility(prices):
    p = pd.Series(list(prices))
    return p.pct_change().dropna().std() * np.sqrt(252)


def beta(stock_prices, market_prices):
    stock = pd.Series(list(stock_prices)).pct_change().dropna()
    market = pd.Series(list(market_prices)).pct_change().dropna()
    return stock.cov(market) / market.var()


def main():
    print(f"Q1 present_value([10, 15, 20], 0.10) = {present_value([10, 15, 20], 0.10):.2f}")
    print(f"Q2 bond_price(1000, 0.04, 10, 0.04) = {bond_price(1000, 0.04, 10, 0.04):.2f}")

    # Q3
    print("\nQ3 10-year bond, 4% coupon, $1000 face:")
    for rate in [0.02, 0.04, 0.0496]:
        print(f"   market rate {rate:.2%} price {bond_price(1000, 0.04, 10, rate):.2f}")

    rates = np.linspace(0, 0.10, 101)
    curve = []
    for rate in rates:
        curve.append(bond_price(1000, 0.04, 10, rate))

    plt.plot(rates * 100, curve)
    plt.title("10-year 4% coupon bond: price vs market rate")
    plt.xlabel("Market rate (%)")
    plt.ylabel("Price ($)")
    plt.savefig("bond_curve.png")
    print("   curve saved to bond_curve.png")

    #Q4
    tickers = ["NVDA", "COST", "CCL"]
    data = {}
    for ticker in tickers + ["SPY"]:
        data[ticker] = yf.Ticker(ticker).history(period="5y")["Close"].dropna()
    df = pd.DataFrame(data).dropna()

    print("\nQ4 trading days over 5 years:")
    for ticker in df.columns:
        print(f"   {ticker} {len(df)}")

    #5
    spy = df["SPY"]
    table = []
    for ticker in tickers + ["SPY"]:
        table.append({"ticker": ticker,
                      "return %": round(annualised_return(df[ticker]) * 100, 1),
                      "volatility %": round(annualised_volatility(df[ticker]) * 100, 1),
                      "beta": round(beta(df[ticker], spy), 2)})
    print("\nQ5")
    print(pd.DataFrame(table).to_string(index=False))

    #   7
    vols = {}
    betas = {}
    for ticker in tickers:
        vols[ticker] = annualised_volatility(df[ticker])
        betas[ticker] = beta(df[ticker], spy)

    print("\nQ7 by volatility: " + " > ".join(sorted(tickers, key=vols.get, reverse=True)))
    print("   by beta: " + " > ".join(sorted(tickers, key=betas.get, reverse=True)))


if __name__ == "__main__":
    main()
