from maxlike.func.tennis import Game, TieBreak, Set, ServeSet
from maxlike.func import Collapse, X, Sum, Linear, Logistic, Scalar
from scipy.special import logit
import maxlike
import numpy as np
import pandas as pd
import os
import unittest
from .numdiff import derivative_errors


class Test(unittest.TestCase):

    def setUp(self):
        # raise on floating point errors, restoring the state afterwards
        # so other test modules are not affected
        self._np_err = np.seterr(all='raise')

    def tearDown(self):
        np.seterr(**self._np_err)
    
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

    def test_derivatives_2d(self):
        # p[i, j]: probability that player i wins a point against player j;
        # the output must stay aligned with (i, j) (regression: the stacking
        # used to transpose the player axes)
        p = np.array([[.5, .3, .6], [.7, .5, .45], [.4, .55, .5]])
        t = np.array([[.5, .2, .65], [.8, .5, .4], [.35, .6, .5]])
        for foo, params in [(Game(), [p]), (TieBreak(), [p]),
                            (Set(), [p, t])]:
            val = np.asarray(foo(params).values)
            for i, j in [(0, 1), (1, 0), (2, 1)]:
                single = [np.array([q[i, j]]) for q in params]
                np.testing.assert_allclose(val[i, j], foo(single).values[0])
            for use_eval in (True, False):
                self.assertEqual(
                    derivative_errors(foo, params, use_eval=use_eval), [])

    def test_derivatives_model(self):
        F = Sum(2)
        F.add(X(), 0, 0)
        F.add(-X(), 0, 1)
        model = Set() @ [
            Collapse(Game(), False),
            Collapse(TieBreak(), False)] @ Logistic() @ F
        a = np.array([.3, -.1, -.2])
        for use_eval in (True, False):
            self.assertEqual(
                derivative_errors(model, [a], use_eval=use_eval), [])

    def test_serve_set(self):
        p = np.array([.3, .5, .62, .7])
        F = Sum(1)
        F.add(X(), 0, 0)
        old = Set() @ [Collapse(Game(), False),
                       Collapse(TieBreak(), False)] @ F
        # no serve advantage: the same as the model without serve
        np.testing.assert_allclose(ServeSet()([p, p]).values,
                                   old([p]).values, atol=1e-14)
        # the 14 outcomes between two players sum to one, whoever serves
        ps, pr = np.array([.66, .6]), np.array([.36, .3])
        both = ServeSet()([ps, pr]).values.sum(-1) + \
            ServeSet()([1 - pr, 1 - ps]).values.sum(-1)
        np.testing.assert_allclose(both, 1)
        # a serve advantage makes sets closer: more tie-breaks
        self.assertGreater(ServeSet()([np.array([.7]), np.array([.4])]
                                      ).values[0, -1],
                           ServeSet()([np.array([.55]), np.array([.55])]
                                      ).values[0, -1])

    def test_serve_set_derivatives(self):
        ps = np.array([[.5, .63, .7], [.6, .55, .66], [.58, .7, .5]])
        pr = np.array([[.5, .33, .4], [.36, .45, .31], [.42, .35, .5]])
        for use_eval in (True, False):
            self.assertEqual(derivative_errors(
                ServeSet(), [ps, pr], use_eval=use_eval), [])

        def logit_model(sign):
            f = Sum(2)
            f.add(X(), 0, 0)
            f.add(-X(), 0, 1)
            f.add(sign * Scalar(), 1, [])
            return f

        model = ServeSet() @ [Logistic() @ logit_model(1.),
                              Logistic() @ logit_model(-1.)]
        a = np.array([.3, -.1, -.2])
        for use_eval in (True, False):
            self.assertEqual(derivative_errors(
                model, [a, .45], use_eval=use_eval), [])


if __name__ == "__main__":
    unittest.main()
