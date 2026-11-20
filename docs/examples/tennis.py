"""
Tennis: player strengths from set scores.

The probability that player i wins a point against player j is
    p[i, j] = logistic(strength[i] - strength[j])
and the probabilities of winning a game, a tie-break and finally a set
with each score (6-0, ..., 7-5, 7-6) follow from p through the exact
recursions of the scoring system (Game, TieBreak, Set). The model is
fitted to the observed set scores (tests/data/data_tennis.csv, ATP
matches) with a multinomial likelihood.

    python docs/examples/tennis.py
"""
import os
import numpy as np
import pandas as pd
import maxlike
from maxlike.func import X, Sum, Linear, Logistic, Collapse
from maxlike.func.tennis import Game, TieBreak, Set

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "tests", "data", "data_tennis.csv")
SCORES = [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (7, 5), (7, 6)]


def load(path=DATA):
    sets = pd.read_csv(path)
    players = sorted(set(sets.player1) | set(sets.player2))
    code = {p: k for k, p in enumerate(players)}
    # N[winner, loser, score]: how often the set ended with that score
    N = np.zeros((len(players), len(players), len(SCORES)))
    for row in sets.itertuples():
        a, b = (row.player1, row.player2) if row.score1 > row.score2 else \
            (row.player2, row.player1)
        score = (max(row.score1, row.score2), min(row.score1, row.score2))
        if score in SCORES:
            N[code[a], code[b], SCORES.index(score)] += 1
    return N, players


def model():
    point = Sum(2)                     # strength[i] - strength[j]
    point.add(X(), 0, 0)
    point.add(-X(), 0, 1)
    # P(i wins the set with each score), from the point probability
    return Set() @ [Collapse(Game(), False), Collapse(TieBreak(), False)] \
        @ Logistic() @ point


def fit(N, players):
    mle = maxlike.NormalizedFinite()
    mle.model = model()
    mle.add_constraint(0, Linear([1]))     # strengths sum to 0
    mle.add_param(pd.Series(0., index=players), name="strength")
    mle.fit(N=N)
    return mle


def main():
    N, players = load()
    mle = fit(N, players)
    print("%d sets, %d players" % (N.sum(), len(players)))

    s = mle.summary().loc["strength"].sort_values("estimate",
                                                   ascending=False)
    print("\nTop 5 (point-level strength, logit scale):")
    print(s[["estimate", "std_error"]].head(5).round(3))

    i, j = players.index(s.index[0]), players.index(s.index[1])
    p_set, se = mle.predict(Collapse(model(), False, True), se=True)
    by_score = mle.predict()
    print("\n%s v %s" % (players[i], players[j]))
    print("  P(%s wins a set) = %.3f +/- %.3f" % (players[i], p_set[i, j],
                                                  se[i, j]))
    print("  most likely scores:")
    probs = {("%d-%d" % sc, players[i]): by_score[i, j, k]
             for k, sc in enumerate(SCORES)}
    probs.update({("%d-%d" % sc, players[j]): by_score[j, i, k]
                  for k, sc in enumerate(SCORES)})
    for (score, winner), p in sorted(probs.items(), key=lambda kv: -kv[1])[:4]:
        print("    %s to %-16s %.3f" % (score, winner, p))
    return mle


if __name__ == "__main__":
    main()
