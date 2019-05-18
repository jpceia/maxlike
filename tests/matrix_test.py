import os
import unittest
import pandas as pd
import numpy as np
import maxlike
from scipy.special import logit
from maxlike.analytics import skellam_cdf_root
from maxlike.preprocessing import prepare_dataframe, prepare_series
from maxlike.func import (X, Linear, Exp, Scalar, 
    Poisson, Sum, Product, CollapseMatrix, MarkovMatrix)


maxlike.tensor.set_dtype(np.float32)


class Test(unittest.TestCase):

    verbose = False
    data_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "matrix")

    def test_logistic_cross(self):
        mle = maxlike.Logistic()
        mle.model = Sum(2)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 0, 1)
        mle.model.add(-Scalar(), 1, [])
        mle.add_constraint([0], Linear([1]))

        # fetch and prepare data
        df = pd.read_csv(
            os.path.join(self.data_folder, "data_proba.csv"),
            index_col=[0, 1])
        df['w'] = df['-1'] + df['1']
        kwargs, _ = prepare_dataframe(df, 'w', '1', {'X': np.sum})
        N = kwargs['N']
        S = kwargs['X']
        u = -logit(S.sum(0) / N.sum(0))
        v = logit(S.sum(1) / N.sum(1))
        a = (u + v) / 2
        h = ((u - v) / 2).mean()

        mle.add_param(a)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        
        a, h = mle.params
        s_a, s_h = mle.std_error()

        df = pd.read_csv(
            os.path.join(self.data_folder, "test_logistic_cross.csv"))
        self.assertAlmostEqual(h,   0.3059389232047434, delta=tol)
        self.assertAlmostEqual(s_h, 0.1053509333552778, delta=tol)
        np.testing.assert_allclose(a,   df['a'],   atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)

    def test_poisson_matrix(self):
        n = 8
        mle = maxlike.Finite(dim=2)

        # fetch and prepare data
        df = pd.read_csv(
            os.path.join(self.data_folder, "data_poisson_matrix.csv"),
            index_col=[0, 1], header=[0, 1]).stack([0, 1])
        kwargs, _ = prepare_series(df)
        N = kwargs['N']

        def EX(u):
            assert u.ndim <= 2
            return (u * np.arange(u.shape[-1])).sum(-1) / u.sum(-1)

        # guess params
        A = EX(N.sum((1, 3)))
        B = EX(N.sum((0, 2)))
        C = EX(N.sum((0, 3)))
        D = EX(N.sum((1, 2)))
        
        log_mean = np.log((A + B + C + D).mean()) / 2
        a = np.log(A + B) - log_mean
        b = log_mean - np.log(C + D)
        h = np.log((A + C).mean()) - np.log((B + D).mean())

        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)
        mle.add_constraint([0, 1], Linear([1, 1]))

        # define functions
        f1 = maxlike.func.Sum(2)
        f1.add(X(), 0, 0)
        f1.add(-X(), 1, 1)
        f1.add(0.5 * Scalar(), 2, [])

        f2 = maxlike.func.Sum(2)
        f2.add(X(), 0, 0)
        f2.add(-X(), 1, 1)
        f2.add(-0.5 * Scalar(), 2, [])

        F1 = Poisson(n) @ Exp() @ f1
        F2 = Poisson(n) @ Exp() @ f2

        F = Product(2, 2)
        F.add(F1, [0, 1, 2], [0, 1], 0)
        F.add(F2, [0, 1, 2], [1, 0], 1)

        mle.model = F

        # calibration
        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(
            os.path.join(self.data_folder, "test_poisson_matrix.csv"))

        self.assertAlmostEqual(h,   0.27852882496320425, delta=tol)
        self.assertAlmostEqual(s_h, 0.05147213154587904, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        #np.testing.assert_allclose(r, df['r_ab'], atol=tol)

    def test_markov_matrix(self): 
        n = 8
        mle = maxlike.Finite(dim=2)

        # fetch and prepare data
        df = pd.read_csv(
            os.path.join(self.data_folder, "data_poisson_matrix.csv"),
            index_col=[0, 1], header=[0, 1]).stack([0, 1])

        kwargs, _ = prepare_series(df)
        N = kwargs['N']

        def EX(u):
            assert u.ndim <= 2
            return (u * np.arange(u.shape[-1])).sum(-1) / u.sum(-1)

        # guess params
        A = EX(N.sum((1, 3)))
        B = EX(N.sum((0, 2)))
        C = EX(N.sum((0, 3)))
        D = EX(N.sum((1, 2)))
        
        log_mean = np.log((A + B + C + D).mean()) / 2
        a = np.log(A + B) - log_mean
        b = log_mean - np.log(C + D)
        h = np.log((A + C).mean()) - np.log((B + D).mean())

        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)
        mle.add_constraint([0, 1], Linear([1, 1]))

        # define functions
        f1 = maxlike.func.Sum(2)
        f1.add(X(), 0, 0)
        f1.add(-X(), 1, 1)
        f1.add(0.5 * Scalar(), 2, [])

        f2 = maxlike.func.Sum(2)
        f2.add(X(), 0, 1)
        f2.add(-X(), 1, 0)
        f2.add(-0.5 * Scalar(), 2, [])

        mle.model = MarkovMatrix(steps=18, size=n) @ \
                    [Exp() @ f1, Exp() @ f2]

        # calibration
        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(
            os.path.join(self.data_folder, "test_markov_matrix.csv"))
        self.assertAlmostEqual(h,   0.28061347212745963, delta=tol)
        self.assertAlmostEqual(s_h, 0.05049593073858663, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        #np.testing.assert_allclose(r, df['r_ab'], atol=tol)

    def test_kullback_leibler(self):
        n = 8
        mle = maxlike.Finite()

        # fetch and prepare data
        df1 = pd.read_csv(
            os.path.join(self.data_folder, "data_proba.csv"),
            index_col=[0, 1])

        kwargs, _ = prepare_series(df1.stack(), {'N': np.sum})

        N = kwargs['N']

        # guess params
        h = (skellam_cdf_root(*(N.sum((0, 1)) / N.sum())[[0, 2]]) *
             np.array([-1, 1])).sum()

        S = N.sum(0) + np.flip(N.sum(1), 1)
        S /= S.sum(1)[:, None]

        s = pd.DataFrame(S[:, [0, 2]]).apply(lambda row:
            pd.Series(skellam_cdf_root(*row), index=['a', 'b']), 1)
        s = np.log(s).sub(np.log(s.mean()), 1)

        mle.add_param(s['a'].values)
        mle.add_param(-s['b'].values)
        mle.add_param(h)
        mle.add_constraint([0, 1], Linear([1, 1]))

        # define functions
        f1 = Sum(2)
        f1.add(X(), 0, 0)
        f1.add(-X(), 1, 1)
        f1.add(0.5 * Scalar(), 2, [])

        f2 = Sum(2)
        f2.add(X(), 0, 0)
        f2.add(-X(), 1, 1)
        f2.add(-0.5 * Scalar(), 2, [])

        F1 = Poisson(n) @ Exp() @ f1
        F2 = Poisson(n) @ Exp() @ f2

        F = Product(2, 2)
        F.add(F1, [0, 1, 2], [0, 1], 1)
        F.add(F2, [0, 1, 2], [1, 0], 0)

        mle.model = CollapseMatrix() @ F

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(
            os.path.join(self.data_folder, "test_kullback_leibler.csv"))
        
        self.assertAlmostEqual(h,   0.27655587703454143, delta=tol)
        self.assertAlmostEqual(s_h, 0.0680302933547584,  delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        #np.testing.assert_allclose(r, df['r_ab'], atol=tol)

    def test_markov_kullback_leibler(self):
        n = 8
        mle = maxlike.Finite()

        # fetch and prepare data
        df1 = pd.read_csv(
            os.path.join(self.data_folder, "data_proba.csv"),
            index_col=[0, 1])
        
        kwargs, _ = prepare_series(df1.stack(), {'N': np.sum})

        N = kwargs['N']

        # guess params
        h = (skellam_cdf_root(*(N.sum((0, 1)) / N.sum())[[0, 2]]) *
             np.array([-1, 1])).sum()

        S = N.sum(0) + np.flip(N.sum(1), 1)
        S /= S.sum(1)[:, None]

        s = pd.DataFrame(S[:, [0, 2]]).apply(lambda row:
            pd.Series(skellam_cdf_root(*row), index=['a', 'b']), 1)
        s = np.log(s).sub(np.log(s.mean()), 1)

        mle.add_param(s['a'].values)
        mle.add_param(-s['b'].values)
        mle.add_param(h)
        mle.add_constraint([0, 1], Linear([1, 1]))

        # define functions
        f1 = maxlike.func.Sum(2)
        f1.add(X(), 0, 0)
        f1.add(-X(), 1, 1)
        f1.add(0.5 * Scalar(), 2, [])

        f2 = maxlike.func.Sum(2)
        f2.add(X(), 0, 1)
        f2.add(-X(), 1, 0)
        f2.add(-0.5 * Scalar(), 2, [])

        mle.model = CollapseMatrix() @ \
                    MarkovMatrix(steps=18, size=n) @ \
                    [Exp() @ f2, Exp() @ f1]

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(
            os.path.join(self.data_folder, "test_markov_kullback_leibler.csv"))
        self.assertAlmostEqual(h,   0.2791943644185061,  delta=tol)
        self.assertAlmostEqual(s_h, 0.06866272277318519, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)


if __name__ == '__main__':
    unittest.main(verbosity=1)
