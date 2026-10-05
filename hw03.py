# Algory QI Hw 3

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

SEED = 42


def simulate_dice(trials, seed=0):
    rng = np.random.default_rng(seed)
    rolled_a_one = 0
    was_four_sided = 0
    for i in range(trials):
        sides = rng.choice([4, 6])
        roll = rng.integers(1, sides + 1)
        if roll == 1:
            rolled_a_one = rolled_a_one + 1
            if sides == 4:
                was_four_sided = was_four_sided + 1
    return was_four_sided / rolled_a_one


def simulate_coins(trials, seed=0):
    rng = np.random.default_rng(seed)
    total = 0
    for i in range(trials):
        heads = rng.integers(0, 2, 3).sum()
        tails = 3 - heads
        total = total + heads * tails
    return total / trials


def p_down(returns):
    down = 0
    for r in returns:
        if r < 0:
            down = down + 1
    return down / len(returns)


def p_down_given_down(returns):
    today_down = 0
    both_down = 0
    for i in range(len(returns) - 1):
        if returns[i] < 0:
            today_down = today_down + 1
            if returns[i + 1] < 0:
                both_down = both_down + 1
    return both_down / today_down


def p_down_given_big_drop(returns, threshold=-0.02):
    big_drops = 0
    then_down = 0
    for i in range(len(returns) - 1):
        if returns[i] < threshold:
            big_drops = big_drops + 1
            if returns[i + 1] < 0:
                then_down = then_down + 1
    return then_down / big_drops


def expected_present_value(cash_flows, rate, survival_prob):
    total = 0
    for t in range(1, len(cash_flows) + 1):
        survives = survival_prob ** (t - 1)  # year 1 is certain
        total = total + cash_flows[t - 1] * survives / (1 + rate) ** t
    return total


def main():
    # 1
    dice = simulate_dice(100_000, seed=SEED)
    print(f"Q1 P(4-sided | rolled a 1) = {dice:.4f}, exact 0.6, difference {dice - 0.6:+.4f}")

     #Q2
    coins = simulate_coins(100_000, seed=SEED)
    print(f"Q2 expected three-coin payout = {coins:.4f}, exact 1.5")

    #3
    print("\nQ3 payout estimate as trials go up:")
    counts = [100, 1_000, 10_000, 100_000]
    estimates = []
    for n in counts:
        estimates.append(simulate_coins(n, seed=SEED))
        print(f"   {n} trials: {estimates[-1]:.4f}")

    plt.plot(counts, estimates, marker="o", label="simulated")
    plt.axhline(1.5, color="red", linestyle="--", label="exact 1.5")
    plt.xscale("log")
    plt.title("Three-coin payout: estimate vs number of trials")
    plt.xlabel("Trials")
    plt.ylabel("Estimated payout")
    plt.legend()
    plt.savefig("coin_convergence.png")
    print("   chart saved to coin_convergence.png")

    # q4
    spy = yf.Ticker("SPY").history(period="10y")["Close"].dropna()
    returns = list(spy.pct_change().dropna())
    print(f"\nQ4 trading days: {len(spy)}")
    print(f"   mean daily return: {np.mean(returns):.5f}")

    #q5
    down_days = 0
    for r in returns:
        if r < 0:
            down_days = down_days + 1

    today_down = 0
    for i in range(len(returns) - 1):
        if returns[i] < 0:
            today_down = today_down + 1

    print(f"\nQ5 P(tomorrow is down) = {p_down(returns):.4f}, from {len(returns)} days")
    print(f"   P(tomorrow is down | today was down) = {p_down_given_down(returns):.4f}, "
          f"from {today_down} down days")

    # Q6
    big_drops = 0
    for i in range(len(returns) - 1):
        if returns[i] < -0.02:
            big_drops = big_drops + 1

    print(f"\nQ6 P(tomorrow is down | today fell more than 2%) = "
          f"{p_down_given_big_drop(returns):.4f}, from only {big_drops} days")

    #7
    print(f"\nQ7 expected_present_value([10, 10, 10], 0.10, 0.5) = "
          f"{expected_present_value([10, 10, 10], 0.10, 0.5):.2f}")


if __name__ == "__main__":
    main()
