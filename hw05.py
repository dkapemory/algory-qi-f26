# QI hw 5

from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TICKER = "SPY"
KINDS = ["long_call", "long_put", "short_call", "short_put"]


def intrinsic(kind, strike, spot):
    if kind == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def extrinsic(kind, strike, spot, premium):
    return premium - intrinsic(kind, strike, spot)


#Q1
def payoff(kind, strike, premium, spot):
    if kind == "long_call":
        return intrinsic("call", strike, spot) - premium
    if kind == "long_put":
        return intrinsic("put", strike, spot) - premium
    if kind == "short_call":
        return premium - intrinsic("call", strike, spot)
    if kind == "short_put":
        return premium - intrinsic("put", strike, spot)
    raise ValueError("unknown kind: " + kind)


#Q2
def payoff_grid(strike, premium, filename):
    spots = np.linspace(strike - 40, strike + 40, 161)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for n in range(len(KINDS)):
        kind = KINDS[n]
        profits = []
        for s in spots:
            profits.append(payoff(kind, strike, premium, s))
        ax = axes[n // 2][n % 2]
        ax.plot(spots, profits)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(strike, color="gray", linestyle="--", label=f"strike {strike}")
        ax.set_title(kind.replace("_", " "))
        ax.set_xlabel("Spot at expiry")
        ax.set_ylabel("Profit per share")
        ax.legend()
    fig.suptitle(f"Payoff at expiry, strike {strike}, premium {premium}")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


#Q3
def days_out(expiry):
    return (date.fromisoformat(expiry) - date.today()).days


def pick_expiry(expiries, min_days):
    for e in expiries:
        if days_out(e) >= min_days:
            return e
    return expiries[-1]


def spot_price(ticker):
    return ticker.history(period="5d")["Close"].iloc[-1]


#Q4
def drop_dead_quotes(calls):
    last_trade = calls["lastTradeDate"].max()
    calls = calls[calls["bid"] > 0]
    calls = calls[calls["lastTradeDate"] >= last_trade - pd.Timedelta(days=7)]
    return calls


def call_values(calls, spot):
    rows = []
    for _, row in calls.iterrows():
        strike = row["strike"]
        mid = (row["bid"] + row["ask"]) / 2
        rows.append({"strike": strike,
                     "mid": mid,
                     "intrinsic": intrinsic("call", strike, spot),
                     "extrinsic": extrinsic("call", strike, spot, mid)})
    return pd.DataFrame(rows)


def nearest_strikes(table, spot, n=10):
    distance = abs(table["strike"] - spot)
    near = table.loc[distance.sort_values().index[:n]]
    return near.sort_values("strike")


def largest_extrinsic(table):
    return table.loc[table["extrinsic"].idxmax()]


def count_above(short, long):
    above = 0
    shared = 0
    for _, row in short.iterrows():
        match = long[long["strike"] == row["strike"]]
        if len(match) > 0:
            shared = shared + 1
            if match["extrinsic"].iloc[0] > row["extrinsic"]:
                above = above + 1
    return above, shared


# q5
def extrinsic_plot(curves, spot, title, filename):
    plt.figure(figsize=(9, 5))
    for label, table in curves:
        plt.plot(table["strike"], table["extrinsic"], marker=".", label=label)
    plt.axvline(spot, color="gray", linestyle="--", label=f"spot {spot:.2f}")
    plt.title(title)
    plt.xlabel("Strike")
    plt.ylabel("Extrinsic value per share")
    plt.legend()
    plt.savefig(filename)
    plt.close()


def main():
    #Q1
    print("Q1")
    for kind in KINDS:
        line = kind + ":"
        for s in [80, 100, 120]:
            line = line + f" spot {s} {payoff(kind, 100, 6, s):.2f},"
        print(line[:-1])

    #Q2
    payoff_grid(100, 6, "payoffs.png")

    #Q3
    t = yf.Ticker(TICKER)
    spot = spot_price(t)
    short_expiry = pick_expiry(t.options, 30)
    chain = t.option_chain(short_expiry)
    calls, puts = chain.calls, chain.puts
    print("\nQ3")
    print(f"ticker: {TICKER}")
    print(f"expiry: {short_expiry}, {days_out(short_expiry)} days out")
    print(f"spot: {spot:.2f}")
    print(f"calls: {len(calls)}, puts: {len(puts)}")

    #Q4
    live = drop_dead_quotes(calls)
    short = call_values(live, spot)
    print("\nQ4")
    print(f"kept {len(live)} of {len(calls)} calls, dropped zero bids and no trade in the last week")
    print(nearest_strikes(short, spot).round(2).to_string(index=False))

    #Q5
    extrinsic_plot([(short_expiry, short)], spot,
                   f"{TICKER} call extrinsic value, {short_expiry} expiry", "extrinsic_short.png")
    top = largest_extrinsic(short)
    low = short.iloc[0]
    high = short.iloc[-1]
    print("\nQ5")
    print(f"largest extrinsic: {top['extrinsic']:.2f} at strike {top['strike']:.0f}")
    print(f"spot: {spot:.2f}")
    print(f"lowest strike {low['strike']:.0f}: extrinsic {low['extrinsic']:.2f}, mid {low['mid']:.2f}")
    print(f"highest strike {high['strike']:.0f}: extrinsic {high['extrinsic']:.2f}")

    #Q6
    long_expiry = pick_expiry(t.options, 182)
    long_calls = t.option_chain(long_expiry).calls
    long_live = drop_dead_quotes(long_calls)
    long = call_values(long_live, spot)
    extrinsic_plot([(short_expiry, short), (long_expiry, long)], spot,
                   f"{TICKER} call extrinsic value, short against long expiry", "extrinsic_both.png")
    top_long = largest_extrinsic(long)
    above, shared = count_above(short, long)
    print("\nQ6")
    print(f"expiry: {long_expiry}, {days_out(long_expiry)} days out, kept {len(long_live)} of {len(long_calls)} calls")
    print(f"largest extrinsic {short_expiry}: {top['extrinsic']:.2f} at strike {top['strike']:.0f}")
    print(f"largest extrinsic {long_expiry}: {top_long['extrinsic']:.2f} at strike {top_long['strike']:.0f}")
    print(f"longer expiry higher at {above} of {shared} shared strikes")


if __name__ == "__main__":
    main()
