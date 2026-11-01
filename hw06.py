# QI hw 6

from datetime import date
from math import log, sqrt, exp, erf

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

TICKER = "SPY"


def intrinsic(kind, strike, spot):
    if kind == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


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


#Q1
def covered_call(entry, strike, premium, spot):
    return (spot - entry) + payoff("short_call", strike, premium, spot)


def protective_put(entry, strike, premium, spot):
    return (spot - entry) + payoff("long_put", strike, premium, spot)


def long_straddle(strike, call_premium, put_premium, spot):
    return payoff("long_call", strike, call_premium, spot) + payoff("long_put", strike, put_premium, spot)


def bull_call_spread(low_strike, low_premium, high_strike, high_premium, spot):
    return payoff("long_call", low_strike, low_premium, spot) + payoff("short_call", high_strike, high_premium, spot)


def summary(spots, profits):
    if profits[-1] > profits[-2]:
        max_profit = "unlimited"
    else:
        max_profit = f"{max(profits):.2f}"
    max_loss = -min(profits)

    breakevens = []
    for i in range(len(spots) - 1):
        a = profits[i]
        b = profits[i + 1]
        if a == 0:
            breakevens.append(spots[i])
        elif a * b < 0:
            breakevens.append(spots[i] - a * (spots[i + 1] - spots[i]) / (b - a))
    return max_profit, max_loss, breakevens


#Q3
def normal_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes(kind, spot, strike, years, rate, vol):
    d1 = (log(spot / strike) + (rate + vol ** 2 / 2) * years) / (vol * sqrt(years))
    d2 = d1 - vol * sqrt(years)
    if kind == "call":
        return spot * normal_cdf(d1) - strike * exp(-rate * years) * normal_cdf(d2)
    return strike * exp(-rate * years) * normal_cdf(-d2) - spot * normal_cdf(-d1)


#Q4
def implied_volatility(price, spot, strike, years, rate, kind):
    low = 0.0001
    high = 5.0
    # no volatility can produce this price
    if price < black_scholes(kind, spot, strike, years, rate, low):
        return None
    if price > black_scholes(kind, spot, strike, years, rate, high):
        return None

    for i in range(100):
        mid = (low + high) / 2
        if black_scholes(kind, spot, strike, years, rate, mid) > price:
            high = mid
        else:
            low = mid
    return (low + high) / 2


#Q5
def last_close(symbol):
    closes = yf.Ticker(symbol).history(period="5d")["Close"]
    return closes.index[-1].date(), closes.iloc[-1]


def chain_ivs(calls, spot, years, rate):
    strikes = []
    ivs = []
    no_solution = 0
    for _, row in calls.iterrows():
        if row["bid"] <= 0:
            continue
        mid = (row["bid"] + row["ask"]) / 2
        iv = implied_volatility(mid, spot, row["strike"], years, rate, "call")
        if iv is None:
            no_solution = no_solution + 1
        else:
            strikes.append(row["strike"])
            ivs.append(iv)
    return strikes, ivs, no_solution


#Q6
def realised_vol(symbol, days=30):
    closes = yf.Ticker(symbol).history(period="3mo")["Close"].iloc[-(days + 1):]
    returns = closes.pct_change().dropna()
    return returns.std() * np.sqrt(252)


def main():
    #Q1
    spot, rate, vol, years = 100, 0.04, 0.25, 0.25
    call_100 = black_scholes("call", spot, 100, years, rate, vol)
    call_110 = black_scholes("call", spot, 110, years, rate, vol)
    put_90 = black_scholes("put", spot, 90, years, rate, vol)
    put_100 = black_scholes("put", spot, 100, years, rate, vol)

    positions = [
        ("covered call, strike 110", lambda s: covered_call(100, 110, call_110, s)),
        ("protective put, strike 90", lambda s: protective_put(100, 90, put_90, s)),
        ("long straddle, strike 100", lambda s: long_straddle(100, call_100, put_100, s)),
        ("bull call spread, 100 and 110", lambda s: bull_call_spread(100, call_100, 110, call_110, s)),
    ]

    print("Q1")
    print(f"spot 100, 3 months, premiums from Black-Scholes at 25% vol: "
          f"call 100 {call_100:.2f}, call 110 {call_110:.2f}, put 90 {put_90:.2f}, put 100 {put_100:.2f}")
    spots = list(np.linspace(0, 200, 20001))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for n in range(len(positions)):
        name, position = positions[n]
        profits = []
        for s in spots:
            profits.append(position(s))
        max_profit, max_loss, breakevens = summary(spots, profits)
        even = ", ".join(f"{b:.2f}" for b in breakevens)
        print(f"{name}: max profit {max_profit}, max loss {max_loss:.2f}, break-even {even}")

        ax = axes[n // 2][n % 2]
        ax.plot(spots[6000:14001], profits[6000:14001])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("Spot at expiry")
        ax.set_ylabel("Profit per share")
    fig.suptitle("Combination payoffs at expiry, spot 100")
    fig.tight_layout()
    fig.savefig("combinations.png")
    plt.close(fig)

    #Q2
    long_leg = black_scholes("call", 100, 100, 0.25, rate, vol)
    short_leg = black_scholes("call", 100, 100, 1 / 52, rate, vol)
    cost = long_leg - short_leg
    print("\nQ2")
    print(f"calendar spread at strike 100: long 3 month call {long_leg:.2f}, "
          f"short 1 week call {short_leg:.2f}, cost {cost:.2f}")
    for later in [90, 100, 110]:
        value = black_scholes("call", later, 100, 0.25 - 1 / 52, rate, vol) - intrinsic("call", 100, later)
        print(f"spot {later} after one week: worth {value:.2f}, profit {value - cost:.2f}")

    #Q3
    call = black_scholes("call", 100, 100, 1, 0.04, 0.25)
    put = black_scholes("put", 100, 100, 1, 0.04, 0.25)
    print("\nQ3")
    print(f"call {call:.2f}, put {put:.2f}")
    print(f"call minus put {call - put:.2f}, S - K*exp(-rT) {100 - 100 * exp(-0.04):.2f}")

    #Q4
    recovered = implied_volatility(call, 100, 100, 1, 0.04, "call")
    print("\nQ4")
    print(f"started at 0.25, recovered {recovered:.4f}")

    #Q5
    t = yf.Ticker(TICKER)
    quote_date, spot = last_close(TICKER)
    rate = last_close("^IRX")[1] / 100
    expiry = None
    for e in t.options:
        if (date.fromisoformat(e) - quote_date).days >= 30:
            expiry = e
            break
    days = (date.fromisoformat(expiry) - quote_date).days
    calls = t.option_chain(expiry).calls
    strikes, ivs, no_solution = chain_ivs(calls, spot, days / 365, rate)

    atm = 0
    for i in range(len(strikes)):
        if abs(strikes[i] - spot) < abs(strikes[atm] - spot):
            atm = i
    vix = last_close("^VIX")[1]

    print("\nQ5")
    print(f"{TICKER} {spot:.2f} at the {quote_date} close, expiry {expiry}, {days} days, rate {rate:.2%}")
    print(f"implied vol for {len(strikes)} calls, {no_solution} with no solution")
    print(f"at the money, strike {strikes[atm]:.0f}: {ivs[atm]:.2%}")
    print(f"lowest strike {strikes[0]:.0f}: {ivs[0]:.2%}, highest strike {strikes[-1]:.0f}: {ivs[-1]:.2%}")
    print(f"VIX: {vix:.2f}")

    plt.figure(figsize=(9, 5))
    plt.plot(strikes, [v * 100 for v in ivs], marker=".")
    plt.axvline(spot, color="gray", linestyle="--", label=f"spot {spot:.2f}")
    plt.title(f"{TICKER} call implied volatility by strike, {expiry} expiry")
    plt.xlabel("Strike")
    plt.ylabel("Implied volatility (%)")
    plt.legend()
    plt.savefig("iv_by_strike.png")
    plt.close()

    #Q6
    realised = realised_vol(TICKER)
    print("\nQ6")
    print(f"realised 30 days: {realised:.2%}, implied at the money: {ivs[atm]:.2%}")
    print(f"implied minus realised: {(ivs[atm] - realised) * 100:.2f} points")


if __name__ == "__main__":
    main()
