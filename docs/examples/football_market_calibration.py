"""
Football: calibrate a score model to market probabilities, then price
markets that are not in the data.

The data (tests/data/data_proba.csv) give, for each match, the market
probabilities of a home win, a draw and an away win (e.g. implied by
bookmaker odds or prediction-market prices). The bivariate Poisson score
model of football_scores.py is collapsed to win/draw/loss and fitted to
those probabilities by minimising the Kullback-Leibler divergence (Finite
likelihood with probabilities in place of counts).

The fitted team strengths then price any function of the score: over/under
goal lines, handicaps, ... consistently with the 1X2 market.

Notes:
- The goal distributions are censored (the last count means "7 or more").
  With a truncated distribution (conditional on fewer than 8 goals, what
  the unnormalised Poisson gives under Finite), strong teams could "win by
  a lot" with absurd scoring rates (over 100 goals a match) whose mass
  piles up at the cap; that degenerate fit even has a better objective.
  Censoring removes it: the fit is the same as on a much wider grid.
- The probabilities are not counts, so the standard errors of a KL fit
  are not sampling errors; they are only useful in relative terms.

    python docs/examples/football_market_calibration.py
"""
import os
import sys
import numpy as np
import pandas as pd
import maxlike
from maxlike.func import Linear, CollapseMatrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from football_scores import score_model  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "tests", "data", "data_proba.csv")


def load(path=DATA):
    df = pd.read_csv(path)
    teams = sorted(set(df.t1) | set(df.t2))
    n = len(teams)
    # P[home, away, outcome], outcome = (home win, draw, away win), the
    # order of CollapseMatrix(); a pair that met more than once adds up
    P = np.zeros((n, n, 3))
    np.add.at(P, (df.t1.values, df.t2.values), df[["1", "0", "-1"]].values)
    names = ["Team %02d" % t for t in teams]
    return P, names, df


def fit(P, names):
    mle = maxlike.Finite()
    mle.model = CollapseMatrix() @ score_model()
    mle.add_constraint([0, 1], Linear([1, 1]))
    mle.add_param(pd.Series(0., index=names), name="attack")
    mle.add_param(pd.Series(0., index=names), name="defence")
    mle.add_param(0., name="home")
    mle.fit(N=P)
    return mle


def main():
    P, names, df = load()
    mle = fit(P, names)

    # how well the model reproduces the market 1X2 probabilities
    fitted = mle.predict(CollapseMatrix() @ score_model())
    played = P.sum(-1) > 0
    market = P[played] / P[played].sum(-1, keepdims=True)
    err = np.abs(fitted[played] - market).max(-1)
    print("%d matches (%d pairs of teams): 1X2 reproduced within %.3f on "
          "average, %.3f for 99%% of the pairs" % (
              len(df), played.sum(), err.mean(), np.quantile(err, .99)))
    a, d, h = [np.ma.getdata(p) for p in mle.get_params()]
    print("scoring rates between %.2f and %.2f goals a match" % (
        np.exp(a[:, None] - d[None, :] - h / 2).min(),
        np.exp(a[:, None] - d[None, :] + h / 2).max()))

    # price markets that are not in the data, for the first match
    row = df.iloc[0]
    i, j = int(row.t1), int(row.t2)
    print("\n%s (home) v %s" % (names[i], names[j]))
    print("  market 1X2: %.3f / %.3f / %.3f" % tuple(P[i, j] / P[i, j].sum()))
    print("  model  1X2: %.3f / %.3f / %.3f" % tuple(fitted[i, j]))
    scores = score_model()
    lines = {
        "over 2.5 goals": (1, 1, -2.5, 1),
        "over 1.5 goals": (1, 1, -1.5, 1),
        "home wins by 2+ (handicap -1.5)": (1, -1, -1.5, 1),
        "home does not win by 2+": (1, -1, -1.5, -1),
    }
    for label, condition in lines.items():
        value, se = mle.predict(CollapseMatrix([condition]) @ scores,
                                se=True)
        print("  %-34s %.3f" % (label, value[i, j, 0]))
    return mle


if __name__ == "__main__":
    main()
