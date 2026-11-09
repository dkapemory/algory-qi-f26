# QI hw 7

import os

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

np.seterr(all="ignore")

LOOKBACKS = (1, 5, 20, 60)
COLUMNS = ["ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "price_vs_ma50", "next_ret"]


#Q1
def features_at(prices, t, lookbacks=LOOKBACKS):
    row = []
    for n in lookbacks:
        row.append(prices[t] / prices[t - n] - 1)
    daily = []
    for i in range(t - 19, t + 1):
        daily.append(prices[i] / prices[i - 1] - 1)
    row.append(np.std(daily, ddof=1) * np.sqrt(252))
    row.append(prices[t] / np.mean(prices[t - 49 : t + 1]) - 1)
    return row


def build_features(prices, lookbacks=LOOKBACKS):
    p = list(prices)
    rows = []
    for t in range(max(lookbacks), len(p) - 1):
        label = p[t + 1] / p[t] - 1  
        rows.append(features_at(p, t, lookbacks) + [label])
    return rows


#Q2
def split_by_time(rows, train=0.6, val=0.2):
    a = int(len(rows) * train)
    b = int(len(rows) * (train + val))
    return rows[:a], rows[a:b], rows[b:]


def up_day_baseline(prices):
    p = list(prices)
    up = 0
    for i in range(1, len(p)):
        if p[i] > p[i - 1]:
            up = up + 1
    return up / (len(p) - 1)


#Q6
def always_hold(window):
    return True


def backtest(prices, strategy, lookback=60):
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
            "fraction_invested": invested / decisions}


def model_rule(model, mean, std):
    def rule(window):
        x = (np.array(features_at(window, len(window) - 1)) - mean) / std
        return model.predict([x])[0] > 0
    return rule


def r2(model, X, y):
    return model.score(X, y)


def main():
    #Q1
    if not os.path.exists("spy.csv"):
        download = yf.Ticker("SPY").history(period="15y")["Close"].dropna()
        download.index = download.index.tz_localize(None)
        download.to_csv("spy.csv")
    spy = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["Close"]
    prices = list(spy)
    rows = build_features(prices)
    dates = list(spy.index[max(LOOKBACKS):-1])
    table = pd.DataFrame(rows, columns=COLUMNS, index=[d.date() for d in dates])

    print("Q1")
    print(f"SPY {len(prices)} trading days, {spy.index[0].date()} to {spy.index[-1].date()}")
    print(f"feature table shape: {table.shape}")
    print("columns:", ", ".join(COLUMNS))
    print(table.head(3).round(4).to_string())

    d = table.index[1000]
    t = 1000 + max(LOOKBACKS)
    by_hand = [spy.pct_change(5).iloc[t],
               spy.pct_change().rolling(20).std().iloc[t] * np.sqrt(252),
               spy.iloc[t] / spy.rolling(50).mean().iloc[t] - 1,
               spy.pct_change().shift(-1).iloc[t]]
    in_table = [table.loc[d, "ret_5"], table.loc[d, "vol_20"], table.loc[d, "price_vs_ma50"], table.loc[d, "next_ret"]]
    print(f"hand check on {d}: ret_5 {in_table[0]:.4f} vs {by_hand[0]:.4f}, vol_20 {in_table[1]:.4f} vs {by_hand[1]:.4f}, "
          f"price_vs_ma50 {in_table[2]:.4f} vs {by_hand[2]:.4f}, next_ret {in_table[3]:.4f} vs {by_hand[3]:.4f}")

    #Q2
    train, val, test = split_by_time(rows)
    a = len(train)
    b = len(train) + len(val)
    print("\nQ2")
    print(f"train: {dates[0].date()} to {dates[a - 1].date()}, {len(train)} rows")
    print(f"validation: {dates[a].date()} to {dates[b - 1].date()}, {len(val)} rows")
    print(f"test: {dates[b].date()} to {dates[-1].date()}, {len(test)} rows")

    X = np.array(rows)[:, :-1]
    y = np.array(rows)[:, -1]
    mean = X[:a].mean(axis=0) 
    std = X[:a].std(axis=0)
    X = (X - mean) / std
    X_train, y_train = X[:a], y[:a]
    X_val, y_val = X[a:b], y[a:b]
    X_test, y_test = X[b:], y[b:]

    #Q3
    models = [("linear regression", LinearRegression()),
              ("ridge regression", Ridge()),
              ("random forest", RandomForestRegressor(random_state=0)),
              ("gradient boosting", GradientBoostingRegressor(random_state=0))]
    print("\nQ3")
    for name, model in models:
        model.fit(X_train, y_train)
        print(f"{name}: train R2 {r2(model, X_train, y_train):.4f}, validation R2 {r2(model, X_val, y_val):.4f}")

    #Q4
    tuning = [("ridge regression", "alpha", [0.1, 1, 10, 100, 1000, 10000, 100000],
               lambda v: Ridge(alpha=v)),
              ("random forest", "max_depth", [1, 2, 3, 4, 6, 8, 12],
               lambda v: RandomForestRegressor(max_depth=v, random_state=0)),
              ("gradient boosting", "n_estimators", [5, 10, 25, 50, 100, 200, 400],
               lambda v: GradientBoostingRegressor(n_estimators=v, random_state=0))]

    lin = LinearRegression().fit(X_train, y_train)
    best = [("linear regression", "none", "-", lin, r2(lin, X_val, y_val))]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    print("\nQ4")
    for n in range(len(tuning)):
        name, param, values, make = tuning[n]
        scores = []
        for v in values:
            model = make(v).fit(X_train, y_train)
            scores.append(r2(model, X_val, y_val))
        k = scores.index(max(scores))
        chosen = make(values[k]).fit(X_train, y_train)
        best.append((name, param, values[k], chosen, scores[k]))
        print(f"{name}: best {param} {values[k]}, validation R2 {scores[k]:.4f}")

        ax = axes[n]
        ax.plot(values, scores, marker="o")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel(param)
        ax.set_ylabel("validation R2")
        if param != "max_depth":
            ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig("tuning.png")
    plt.close(fig)

    #Q5
    name, param, value, model, score = max(best, key=lambda m: m[4])
    predicted = model.predict(X_test)
    accuracy = np.mean((predicted > 0) == (y_test > 0))
    up_days = up_day_baseline(prices[b + max(LOOKBACKS):])
    print("\nQ5")
    print(f"best model: {name}, {param} {value}, validation R2 {score:.4f}")
    print(f"test R2 {r2(model, X_test, y_test):.4f}, correlation of predicted and actual {np.corrcoef(predicted, y_test)[0, 1]:.4f}")
    print(f"right direction {accuracy:.1%} of days, share of up days {up_days:.1%}")

    #Q6
    test_prices = prices[b : ] 
    s = backtest(test_prices, model_rule(model, mean, std), lookback=61)
    h = backtest(test_prices, always_hold, lookback=61)
    print("\nQ6")
    print(f"strategy: annual {s['annualised_return']:.2%}, max drawdown {s['max_drawdown']:.1%}, invested {s['fraction_invested']:.1%}")
    print(f"buy and hold: annual {h['annualised_return']:.2%}, max drawdown {h['max_drawdown']:.1%}, invested 100.0%")

    #Q7
    print("\nQ7")
    if hasattr(model, "feature_importances_"):
        weights = model.feature_importances_
        print(f"{name} feature importances:")
    else:
        weights = model.coef_
        print(f"{name} coefficients on standardised features:")
    order = sorted(range(len(weights)), key=lambda i: -abs(weights[i]))
    for i in order:
        print(f"{COLUMNS[i]}: {weights[i]:.6f}")


if __name__ == "__main__":
    main()
