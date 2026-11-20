"""
Football: full score distribution of a match, and the probabilities of
the usual betting markets derived from it.

Home and away goals are Poisson with rates
    exp(attack[home] - defence[away] + home / 2)
    exp(attack[away] - defence[home] - home / 2)
and the model is the joint distribution of the score (x, y), fitted to the
observed scores with a Finite (multinomial) likelihood over the 8 x 8
score grid (scores capped at 7). Win/draw/loss and over/under 2.5 goals
are then functions of the score matrix (CollapseMatrix), predicted with
delta-method standard errors.

    python docs/examples/football_scores.py
"""
import os
import numpy as np
import pandas as pd
import maxlike
from maxlike.func import (
    X, Scalar, Sum, Product, Poisson, Exp, Linear, CollapseMatrix)
from maxlike.preprocessing import prepare_series

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "tests", "data", "data2.csv")
MAX_GOALS = 8


def rates():
    """log scoring rates of the home and the away team"""
    home_rate = Sum(2)
    home_rate.add(X(), 0, 0)               # attack of the home team
    home_rate.add(-X(), 1, 1)              # defence of the away team
    home_rate.add(0.5 * Scalar(), 2, [])   # home advantage
    away_rate = Sum(2)
    away_rate.add(X(), 0, 0)
    away_rate.add(-X(), 1, 1)
    away_rate.add(-0.5 * Scalar(), 2, [])
    return home_rate, away_rate


def score_model(size=MAX_GOALS, normalised=False):
    """
    P(home scores x, away scores y) on the (home, away, x, y) grid.

    maxlike's Poisson is lambda^x / x! (without exp(-lambda)): enough to fit
    a Finite likelihood, which normalises over the grid. normalised=True
    adds the exp(-lambda) factors, for actual probabilities.
    """
    home_rate, away_rate = rates()
    score = Product(2, 2)
    # home goals: teams in (home, away) order, distribution axis 0 (x)
    score.add(Poisson(size) @ Exp() @ home_rate, [0, 1, 2], [0, 1], 0)
    # away goals: teams swapped, distribution axis 1 (y)
    score.add(Poisson(size) @ Exp() @ away_rate, [0, 1, 2], [1, 0], 1)
    if normalised:
        score.add(Exp() @ (-1. * (Exp() @ home_rate)), [0, 1, 2], [0, 1])
        score.add(Exp() @ (-1. * (Exp() @ away_rate)), [0, 1, 2], [1, 0])
    return score


def fit(path=DATA):
    matches = pd.read_csv(path)
    matches[["g1", "g2"]] = np.minimum(matches[["g1", "g2"]], MAX_GOALS - 1)
    teams = ["Team %02d" % t for t in sorted(set(matches.t1) | set(matches.t2))]
    # counts of each score for each (home, away) pair
    counts = matches.assign(n=1).set_index(["t1", "t2", "g1", "g2"])["n"]
    data, _ = prepare_series(counts, add_axis={"g1": np.arange(MAX_GOALS),
                                               "g2": np.arange(MAX_GOALS)})
    mle = maxlike.Finite(dim=2)
    mle.model = score_model()
    mle.add_constraint([0, 1], Linear([1, 1]))
    mle.add_param(pd.Series(0., index=teams), name="attack")
    mle.add_param(pd.Series(0., index=teams), name="defence")
    mle.add_param(0., name="home")
    mle.fit(**data)
    return mle, teams


def main():
    mle, teams = fit()
    s = mle.summary()
    strength = s.loc["attack", "estimate"] + s.loc["defence", "estimate"]
    home, away = strength.idxmax(), strength.sort_values().index[len(teams) // 2]
    i, j = teams.index(home), teams.index(away)

    # probabilities on a larger grid (scores up to 19), so that no
    # mass is lost; markets are functions of the score matrix, predicted
    # with standard errors
    scores = score_model(size=20, normalised=True)
    markets = {
        "1X2": (CollapseMatrix(), ["home win", "draw", "away win"]),
        "goals": (CollapseMatrix([(1, 1, -2.5, 1), (1, 1, -2.5, -1)]),
                  ["over 2.5", "under 2.5"]),
    }
    print("%s (home) v %s" % (home, away))
    for name, (collapse, labels) in markets.items():
        value, se = mle.predict(collapse @ scores, se=True)
        for k, label in enumerate(labels):
            print("  %-10s %.3f +/- %.3f" % (label, value[i, j, k],
                                           se[i, j, k]))

    probs = mle.predict(scores)[i, j]
    x, y = np.unravel_index(np.argmax(probs), probs.shape)
    print("  most likely score %d-%d (probability %.3f); total mass %.6f" % (
        x, y, probs[x, y], probs.sum()))
    return mle


if __name__ == "__main__":
    main()
