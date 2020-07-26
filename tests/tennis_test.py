from maxlike.func.tennis import Game, TieBreak, Set
from maxlike.func import Collapse, X, Sum, Linear, Logistic
from scipy.special import logit
import maxlike
import numpy as np
import pandas as pd
import os
import unittest
np.seterr(all='raise')


class Test(unittest.TestCase):
    
    verbose = False
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    def test_set(self):
        df1 = pd.read_csv(os.path.join(self.data_folder, "data_tennis.csv"))
        df2 = df1.iloc[:, [1, 0, 3, 2]].copy()
        df2.columns = ["player1", "player2", "score1", "score2"]
        df = pd.concat([df1, df2], axis=0).set_index(["player1", "player2"])
        players = sorted(df.index.levels[0])

        for s in [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (7, 5), (7, 6)]:
            df[s] = (df["score1"] == s[0]) & (df["score2"] == s[1])

        del df["score1"]
        del df["score2"]

        df = df[df.sum(1) == 1].groupby(level=[0, 1]).sum().\
           reindex(pd.MultiIndex.from_product([players, players]), fill_value=0)

        N = df.values.reshape((len(players), len(players), 7))
        w = np.array([
            0.845968173802841,
            0.651691199193479,
            0.5948742493867993,
            0.5581538754948198,
            0.5309672297329231,
            0.5140352691525634,
            0.5044100735840071,
        ])

        N_ = N.sum(-1)
        X_ = (w * N).sum(-1)
        X_ = X_ + (N_ - X_).transpose()
        N_ = N_ + N_.transpose()
        a_guess = -logit(X_.sum(0) / N_.sum(0))

        mle = maxlike.NormalizedFinite()
        F = Sum(2)
        F.add(X(), 0, 0)
        F.add(-X(), 0, 1)

        mle.model = Set() @ [
            Collapse(Game(), False),
            Collapse(TieBreak(), False)] @ \
            Logistic() @ F

        mle.add_constraint(0, Linear([1]))
        mle.add_param(a_guess)

        tol = 1e-8
        mle.fit(N=N, verbose=self.verbose)
        a = mle.get_params()[0]
        s_a = mle.std_error()[0]

        df = pd.read_csv(
            os.path.join(self.data_folder, "test_tennis_set.csv"),
            index_col=[0])
        np.testing.assert_allclose(a,   df['a'],    atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'],  atol=tol)


if __name__ == "__main__":
    unittest.main()
