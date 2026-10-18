# QI hw 4

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

LOOKBACK = 60
SPLIT_DATE = "2011-09-09"


#Q1
def load_and_split(split_date=SPLIT_DATE, prices=None, warmup=LOOKBACK):
    if prices is None:
        prices = yf.Ticker("^SP500TR").history(start="1990-01-01")["Close"].dropna()

    train = prices.loc[:split_date]
    test = prices.iloc[len(train) - warmup:]
    return train, test


# Q2, Q4
def always_hold(window):
    return True


def always_cash(window):
    return False


def make_random_hold(p=0.7, seed=0):
    rng = np.random.default_rng(seed)

    def random_hold(window):
        return bool(rng.random() < p)

    return random_hold


# q5 
def five_day_reversal(window):
    five_day_return = window[-1] / window[-6] - 1
    if five_day_return > 0:
        return False
    return True


# q3
def backtest(prices, strategy, lookback=LOOKBACK):
    p = list(prices)
    value = 1.0
    values = [1.0]
    invested = 0
    decisions = 0

    for i in range(lookback - 1, len(p) - 1):
        window = p[i - lookback + 1 : i + 1]
        decisions = decisions + 1
        if strategy(window):
            invested = invested + 1
            value = value * (p[i + 1] / p[i])
        values.append(value)

    peak = values[0]
    max_drawdown = 0.0
    for v in values:
        if v > peak:
            peak = v
        fall = v / peak - 1
        if fall < max_drawdown:
            max_drawdown = fall

    days = len(values) - 1
    return {"total_return": value - 1,
            "annualised_return": value ** (252 / days) - 1,
            "max_drawdown": max_drawdown,
            "fraction_invested": invested / decisions,
            "values": values}


#q7
def my_strategy(window):
    average = np.mean(window)
    five_day_return = window[-1] / window[-6] - 1
    if window[-1] <= average and five_day_return > 0:
        return False
    return True


#q6
def volatility(window):
    daily = np.diff(np.log(window[-21:]))
    return np.std(daily) * np.sqrt(252)


def momentum_59(window):
    return window[-1] > window[0]


def momentum_30(window):
    return window[-1] > window[-31]


def momentum_15(window):
    return window[-1] > window[-16]


def above_average_60(window):
    return window[-1] > np.mean(window)


def above_average_30(window):
    return window[-1] > np.mean(window[-30:])


def above_average_15(window):
    return window[-1] > np.mean(window[-15:])


def slope_60(window):
    slope = np.polyfit(range(60), window, 1)[0]
    return slope > 0


def slope_30(window):
    slope = np.polyfit(range(30), window[-30:], 1)[0]
    return slope > 0


def low_vol_20(window):
    return volatility(window) < 0.20


def low_vol_25(window):
    return volatility(window) < 0.25


def near_high_10(window):
    return window[-1] >= 0.90 * max(window)


def near_high_5(window):
    return window[-1] >= 0.95 * max(window)


def crossover(window):
    return np.mean(window[-10:]) > np.mean(window[-30:])


def above_average_and_down(window):
    return window[-1] > np.mean(window) and window[-1] / window[-6] - 1 <= 0


def above_average_and_vol_20(window):
    return window[-1] > np.mean(window) and volatility(window) < 0.20


def above_average_and_vol_25(window):
    return window[-1] > np.mean(window) and volatility(window) < 0.25


VARIANTS = [
    ("above price 59 days ago", momentum_59),
    ("above price 30 days ago", momentum_30),
    ("above price 15 days ago", momentum_15),
    ("above 60 day average", above_average_60),
    ("above 30 day average", above_average_30),
    ("above 15 day average", above_average_15),
    ("60 day slope positive", slope_60),
    ("30 day slope positive", slope_30),
    ("volatility under 20%", low_vol_20),
    ("volatility under 25%", low_vol_25),
    ("within 10% of 60 day high", near_high_10),
    ("within 5% of 60 day high", near_high_5),
    ("10 day average above 30 day average", crossover),
    ("above 60 day average and 5 day return not positive", above_average_and_down),
    ("above 60 day average and volatility under 20%", above_average_and_vol_20),
    ("above 60 day average and volatility under 25%", above_average_and_vol_25),
    ("hold unless below 60 day average and up over 5 days", my_strategy),
]


def rebased_plot(prices, strategy, title, filename):
    strat = backtest(prices, strategy)["values"]
    index = backtest(prices, always_hold)["values"]
    plt.figure()
    plt.plot(index, label="index (buy and hold)")
    plt.plot(strat, label="strategy")
    plt.title(title)
    plt.xlabel("Trading days since the first day traded")
    plt.ylabel("Value of 1 invested")
    plt.legend()
    plt.savefig(filename)
    plt.close()


def show(name, r):
    print(f"{name}: total {r['total_return']:.1%}, annual {r['annualised_return']:.2%}")


def main():
    # Q1
    train, test = load_and_split()
    print("Q1")
    print(f"train: {len(train)} days, mean price {train.mean():.2f}")
    print(f"test: {len(test)} days, mean price {test.mean():.2f}")

    #Q3, q4, q5
    index = backtest(train, always_hold)
    total = train.iloc[-1] / train.iloc[LOOKBACK - 1] - 1
    annual = (1 + total) ** (252 / (len(train) - LOOKBACK)) - 1

    print("\nQ3")
    show("strategy", index)
    print(f"index: total {total:.1%}, annual {annual:.2%}")
    rebased_plot(train, always_hold, "Check: always hold sits on the index", "backtest_check.png")

    print("\nQ4")
    show("always cash", backtest(train, always_cash))
    show("random 70%", backtest(train, make_random_hold(0.7, seed=0)))

    print("\nQ5")
    show("5 day reversal", backtest(train, five_day_reversal))
    show("index", index)

    # Q6
    print("\nQ6")
    beat = 0
    for label, strategy in VARIANTS:
        r = backtest(train, strategy)
        print(f"{label}: {r['annualised_return']:.2%}")
        if r["annualised_return"] > index["annualised_return"]:
            beat = beat + 1
    print(f"index: {index['annualised_return']:.2%}")
    print(f"{beat} of {len(VARIANTS)} beat the index")

    # Q7
    final = backtest(train, my_strategy)
    print("\nQ7")
    print("hold unless below the 60 day average and up over the last 5 days")
    print(f"train: annual {final['annualised_return']:.2%}, max drawdown {final['max_drawdown']:.1%}, "
          f"invested {final['fraction_invested']:.0%}")
    print(f"index: annual {index['annualised_return']:.2%}, max drawdown {index['max_drawdown']:.1%}")
    rebased_plot(train, my_strategy, "Final strategy against the index, training set",
                 "strategy_train.png")

    #q8
    s = backtest(test, my_strategy)
    i = backtest(test, always_hold)
    print("\nQ8")
    print(f"strategy: total {s['total_return']:.1%}, annual {s['annualised_return']:.2%}, "
          f"max drawdown {s['max_drawdown']:.1%}, invested {s['fraction_invested']:.1%}")
    print(f"index: total {i['total_return']:.1%}, annual {i['annualised_return']:.2%}, "
          f"max drawdown {i['max_drawdown']:.1%}")
    rebased_plot(test, my_strategy, "Final strategy against the index, test set",
                 "strategy_test.png")

    #Q9
    window = list(test)[-LOOKBACK:]
    decision = "hold" if my_strategy(window) else "cash"
    print("\nQ9")
    print(f"{decision}, last close {test.index[-1].date()}")


if __name__ == "__main__":
    main()
