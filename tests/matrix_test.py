import os
import unittest
import pandas as pd
import numpy as np
import maxlike
from scipy.special import logit
from maxlike.skellam import skellam_cdf_root
from maxlike.preprocessing import prepare_dataframe, prepare_series
from maxlike.func import (
    X, Linear, Exp, Scalar, Sum, Product,
    Poisson, NegativeBinomial,
    CollapseMatrix, MarkovVector, MarkovMatrix)


class Test(unittest.TestCase):

    verbose = False
    data_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data")

    def test_logistic_cross(self):
        mle = maxlike.Logistic()
        mle.model = Sum(2)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 0, 1)
        mle.model.add(-Scalar(), 1, [])
        mle.add_constraint([0], Linear(1))

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
        
        a, h = mle.get_params()
        s_a, s_h = mle.std_error()

        df = pd.read_csv(
            os.path.join(self.data_folder, "test_logistic_cross.csv"))
        self.assertAlmostEqual(h,   0.30593893749482887, delta=tol)
        self.assertAlmostEqual(s_h, 0.1053509333552778, delta=tol)
        np.testing.assert_allclose(a,   df['a'],   atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)

    def test_poisson_matrix(self):
        n = 8
        mle = maxlike.Finite(dim=2)

        # fetch and prepare data
        df = pd.read_csv(os.path.join(self.data_folder, "data2.csv"))
        df[['g1', 'g2']] = np.minimum(df[['g1', 'g2']], n - 1)

        h = np.log(df.g1.mean()) - np.log(df.g2.mean())
        log_mean = np.log(df.mean().mean()) / 2
        a = df.groupby('t1').g1.mean() + df.groupby('t2').g2.mean()
        b = df.groupby('t1').g2.mean() + df.groupby('t2').g1.mean()
        a = np.log(a) - log_mean
        b = log_mean - np.log(b)

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

        df['dummy'] = 1
        df = df.set_index(['t1', 't2', 'g1', 'g2'])['dummy']
        axis = {'g1': np.arange(n), 'g2': np.arange(n)}
        kwargs, _ = prepare_series(df, add_axis=axis)
        N = kwargs['N']

        # calibration
        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.get_params()
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
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

    def test_markov_matrix(self): 
        n = 8
        mle = maxlike.Finite(dim=2)

        # fetch and prepare data
        df = pd.read_csv(os.path.join(self.data_folder, "data2.csv"))
        df[['g1', 'g2']] = np.minimum(df[['g1', 'g2']], n - 1)

        h = np.log(df.g1.mean()) - np.log(df.g2.mean())
        log_mean = np.log(df.mean().mean()) / 2
        a = df.groupby('t1').g1.mean() + df.groupby('t2').g2.mean()
        b = df.groupby('t1').g2.mean() + df.groupby('t2').g1.mean()
        a = np.log(a) - log_mean
        b = log_mean - np.log(b)

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

        df['dummy'] = 1
        df = df.set_index(['t1', 't2', 'g1', 'g2'])['dummy']
        axis = {'g1': np.arange(n), 'g2': np.arange(n)}
        kwargs, _ = prepare_series(df, add_axis=axis)
        N = kwargs['N']

        # calibration
        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(
            os.path.join(self.data_folder, "test_markov_matrix.csv"))
        self.assertAlmostEqual(h,   0.2806134582870986, delta=tol)
        self.assertAlmostEqual(s_h, 0.05049593073858663, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

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
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(
            os.path.join(self.data_folder, "test_kullback_leibler.csv"))
        
        self.assertAlmostEqual(h,   0.276555903327557, delta=tol)
        self.assertAlmostEqual(s_h, 0.0680302933547584, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

    def test_markov_vector(self):
        n = 8
        mle = maxlike.Finite(dim=1)

        # fetch and prepare data
        df = pd.read_csv(os.path.join(self.data_folder, "data2.csv"))

        h = np.log(df.g1.mean()) - np.log(df.g2.mean())
        log_mean = np.log(df.mean().mean()) / 2
        a = df.groupby('t1').g1.mean() + df.groupby('t2').g2.mean()
        b = df.groupby('t1').g2.mean() + df.groupby('t2').g1.mean()
        a = np.log(a) - log_mean
        b = log_mean - np.log(b)

        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)
        mle.add_constraint([0, 1], Linear([1, 1]))

        # define functions
        f1 = maxlike.func.Sum(2)
        f1.add(X(), 0, 0)
        f1.add(-X(), 1, 1)
        f1.add(0.5 * Scalar(), 2, None)

        f2 = maxlike.func.Sum(2)
        f2.add(X(), 0, 1)
        f2.add(-X(), 1, 0)
        f2.add(-0.5 * Scalar(), 2, None)

        mle.model = MarkovVector(steps=18, size=n) @ [Exp() @ f1, Exp() @ f2]

        df['d'] = np.clip(df['g1'] - df['g2'], 1 - n, n - 1)
        df = df.set_index(['t1', 't2', 'd'])
        axis = {'d': np.arange(2 * n - 1) - n + 1}
        kwargs, _ = prepare_series(df['g1'], {'N': np.size}, add_axis=axis)

        tol = 1e-8
        mle.fit(**kwargs, verbose=0, tol=tol)

        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(
            os.path.join(self.data_folder, "test_markov_vector.csv"))
        
        self.assertAlmostEqual(h,   0.3237922149456103, delta=tol)
        self.assertAlmostEqual(s_h, 0.05713304434379314, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

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
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(
            os.path.join(self.data_folder, "test_markov_kullback_leibler.csv"))
        self.assertAlmostEqual(h,   0.2791943644185061, delta=tol)
        self.assertAlmostEqual(s_h, 0.06866272277318519, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)


if __name__ == '__main__':
    unittest.main(verbosity=1)
