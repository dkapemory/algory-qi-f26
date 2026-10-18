# algory-qi-f26

Coursework for Algory Capital's Quantitative Investing training program, Fall 2026.
Each assignment is one self-contained Python file that prints its own answers.

| File | What it does |
| --- | --- |
| `hw01.py` | Pulls a year of daily closes for SCHW and SPY, compares return and annualised volatility, and writes the data out for charting. |
| `hw02.py` | Present value and bond pricing from scratch, plus annualised return, volatility and beta against SPY for three tickers over five years. |
| `hw03.py` | Two Monte Carlo simulations (conditional probability and an expected payout), then tests whether SPY daily returns are independent by counting directly. Also extends present value to a yearly survival probability. |
| `hw04.py` | A walk-forward backtester and a strategy built on 1990 to 2011 data, then tested once on 2011 to 2026. |

## hw04.py, the backtester

The piece worth reusing. It walks forward one trading day at a time over the
S&P 500 total return index (`^SP500TR`) and never lets a strategy see the day it
is predicting.

**The convention.** On day `i` the strategy receives `prices[i-59 : i+1]`, the 60
closes ending with today's. If it returns `True` the portfolio earns
`prices[i+1] / prices[i] - 1`, otherwise zero. Today's close is known when the
decision is made, so withholding it would throw away a day of information.
Tomorrow's close is what is being predicted, so reading it is the bug the whole
exercise is about.

**How it is verified.** Three controls, all of which have answers known in
advance:

- always hold must match buy and hold to the cent over the days actually traded
- always cash must return exactly 0.0%
- a probe strategy run over a strictly increasing series must never be shown the
  final price

`backtest()` returns total return, annualised return (252 trading days,
compounded), maximum drawdown measured from the running peak of portfolio value,
and the fraction of days invested.

**Data split.** Training is 1990-01-02 to 2011-09-09. The test set begins 60
trading days before the split so the lookback window is already full on the
first out-of-sample day. Cut at a single date with no overlap and the
out-of-sample period silently starts three months late.

## The result, stated honestly

Seventeen strategy variants were tried. Every one is printed with its training
return, because the number of things tried is part of the result.

One of the seventeen beat buy and hold in training: hold the index unless it is
below its 60-day moving average and has risen over the last five days. Training
return was 8.65% a year against 8.10% for the index, with a 38.7% maximum
drawdown against 55.3%.

Out of sample it lost, returning 12.59% a year against the index's 15.47%. The
reason is visible in the data: the days it sat out averaged -0.006% in training
but +0.136% out of sample, against a +0.063% average for all test days. The rule
was avoiding slightly bad days in one era and the best days in the next, so its
premise inverted rather than simply decaying.

The drawdown reduction did survive out of sample, 26.9% against 33.8%. That is
the only part of the result worth keeping.

## Running it

```
pip install yfinance pandas numpy matplotlib
python3 hw04.py
```

The self-check, `check_hw04.py`, lives in the course's
[starter repository](https://github.com/WilliamANeild/algory-qi-f26-starter).
