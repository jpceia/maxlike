import os
import unittest
import pandas as pd
import numpy as np
import sys; sys.path.insert(0, "..")
import maxlike
from scipy.special import logit
from maxlike.skellam import skellam_cdf_root
from maxlike.preprocessing import prepare_dataframe, prepare_series, df_count
from maxlike.func import (
    X, Vector, Linear, Quadratic, Exp, Log, Constant, Scalar, 
    Poisson, NegativeBinomial, Sum, Product)


np.seterr(all='raise', under='ignore')
data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class Test(unittest.TestCase):

    verbose = 0

    def test_poisson(self):
        mle = maxlike.Poisson()
        mle.model = Sum(3)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 1, 1)
        mle.model.add(Vector(np.arange(2) - .5), 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))

        g = pd.read_csv(os.path.join(data_folder, "data1.csv"), index_col=[0, 1, 2])['g']
        kwargs, _ = prepare_series(g, {'N': np.size, 'X': np.sum})

        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean()) / 2
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean().map(np.log)
        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_poisson.csv"))
        self.assertAlmostEqual(h,   0.2749716224110226, delta=tol)
        self.assertAlmostEqual(s_h, 0.05113269003880641, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

        aic = mle.akaine_information_criterion()
        bic = mle.bayesian_information_criterion()
        self.assertAlmostEqual(aic, 3.426092788838951, delta=tol)
        self.assertAlmostEqual(bic, 3.590817107890838, delta=tol)

    def test_poisson1(self):
        mle = maxlike.Poisson()
        mle.model = Sum(3)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 1, 1)
        mle.model.add(Vector(np.arange(2) - .5), 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))

        g = pd.read_csv(os.path.join(data_folder, "data1.csv"), index_col=[0, 1, 2])['g']
        kwargs, _ = prepare_series(g, {'N': np.size, 'X': np.sum})

        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean()) / 2
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean().map(np.log)
        a_fix = 2
        b_fix = 3
        a[a_fix] = 1
        b[b_fix] = -1
        mle.add_param(a, np.arange(a.size) == a_fix)
        mle.add_param(b, np.arange(b.size) == b_fix)
        mle.add_param(h, False)

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_poisson1.csv"))
        self.assertAlmostEqual(h,   0.2541711117084739,  delta=tol)
        self.assertAlmostEqual(s_h, 0.04908832460966403, delta=tol)
        np.testing.assert_allclose(a[~a.mask], df.loc[~a.mask, 'a'], atol=tol)
        np.testing.assert_allclose(b[~b.mask], df.loc[~b.mask, 'b'], atol=tol)
        np.testing.assert_allclose(s_a[~a.mask], df.loc[~a.mask, 's_a'], atol=tol)
        np.testing.assert_allclose(s_b[~b.mask], df.loc[~b.mask, 's_b'], atol=tol)
        np.testing.assert_allclose(r[~(a.mask|b.mask)],
                                   df.loc[~(a.mask|b.mask), 'r_ab'], atol=tol)

        aic = mle.akaine_information_criterion()
        bic = mle.bayesian_information_criterion()
        self.assertAlmostEqual(aic, 3.6204860853956444, delta=tol)
        self.assertAlmostEqual(bic, 3.7852104044475317, delta=tol)

    def test_poisson2(self):
        mle = maxlike.Poisson()
        mle.model = Sum(3)
        s = Sum(2)
        s.add(X(), 0, 0)
        s.add(-X(), 1, 1)
        k = Constant(np.arange(2) - .5)
        f_h = Product(1).add(k, None, 0).add(Scalar(), 0, None)
        mle.model.add(s, [0, 1], [0, 1])
        mle.model.add(f_h, 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))

        g = pd.read_csv(os.path.join(data_folder, "data1.csv"), index_col=[0, 1, 2])['g']
        kwargs, _ = prepare_series(g, {'N': np.size, 'X': np.sum})

        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean()) / 2
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean().map(np.log)
        a_fix = 2
        b_fix = 3
        a[a_fix] = 1
        b[b_fix] = -1
        mle.add_param(a, np.arange(a.size) == a_fix)
        mle.add_param(b, np.arange(b.size) == b_fix)
        mle.add_param(h, False)

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_poisson1.csv"))
        self.assertAlmostEqual(h,   0.2541711117084739,  delta=tol)
        self.assertAlmostEqual(s_h, 0.04908832460966403, delta=tol)
        np.testing.assert_allclose(a[~a.mask], df.loc[~a.mask, 'a'], atol=tol)
        np.testing.assert_allclose(b[~b.mask], df.loc[~b.mask, 'b'], atol=tol)
        np.testing.assert_allclose(s_a[~a.mask], df.loc[~a.mask, 's_a'], atol=tol)
        np.testing.assert_allclose(s_b[~b.mask], df.loc[~b.mask, 's_b'], atol=tol)
        np.testing.assert_allclose(r[~(a.mask|b.mask)],
                                   df.loc[~(a.mask|b.mask), 'r_ab'], atol=tol)

    def test_poisson3(self):
        mle = maxlike.Poisson()
        s = Sum(2).add(X(), 0, 0).add(-X(), 1, 1)

        # s_ij = a_i - b_j
        s_diff = Sum(2)
        s_diff.add(s, [0, 1], [0, 1])
        s_diff.add(-s, [0, 1], [1, 0])

        # s_diff_ij = a_i - a_j + b_i - b_j
        hs = Product(2)
        hs.add(Scalar(), 2, None)
        hs.add(s_diff, [0, 1], [0, 1])
        h_diff = Sum(2)
        h_diff.add(Scalar(), 2, None)
        h_diff.add(hs, [0, 1, 3], [0, 1])
        H = Product(3)
        H.add(Constant(np.arange(2) - .5), [], 2)
        H.add(h_diff, [0, 1, 2, 3], [0, 1])

        F = Sum(3)
        F.add(s, [0, 1], [0, 1])
        F.add(H, [0, 1, 2, 3], [0, 1, 2])

        mle.add_constraint([0, 1], Linear([1, 1]))

        mle.model = F

        g = pd.read_csv(os.path.join(data_folder, "data1.csv"), index_col=[0, 1, 2])['g']
        kwargs, _ = prepare_series(g, {'N': np.size, 'X': np.sum})

        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        h1 = 0
        log_mean = np.log(g.mean()) / 2
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean().map(np.log)

        a_fix = 2
        b_fix = 3
        a[a_fix] = 1
        b[b_fix] = -1

        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)
        mle.add_param(h1)

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h, h1 = mle.get_params()
        s_a, s_b, s_h, s_h1 = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_poisson3.csv"))
        self.assertAlmostEqual(h,    0.23613272896129883, delta=tol)
        self.assertAlmostEqual(s_h,  0.055051545352072025, delta=tol)
        self.assertAlmostEqual(h1,   0.13188676189444215, delta=tol)
        self.assertAlmostEqual(s_h1, 0.07187480574001189, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

        aic = mle.akaine_information_criterion()
        bic = mle.bayesian_information_criterion()
        self.assertAlmostEqual(aic, 3.4249498108457352, delta=tol)
        self.assertAlmostEqual(bic, 3.5937335954880614, delta=tol)

    def test_poisson_broyden(self):
        mle = maxlike.Poisson()
        mle.model = Sum(3)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 1, 1)
        mle.model.add(Vector(np.arange(2) - .5), 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))

        g = pd.read_csv(os.path.join(data_folder, "data1.csv"), index_col=[0, 1, 2])['g']
        kwargs, _ = prepare_series(g, {'N': np.size, 'X': np.sum})

        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean()) / 2
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean().map(np.log)
        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(**kwargs, method="broyden", verbose=self.verbose)
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_poisson_broyden.csv"))
        self.assertAlmostEqual(h,   0.2749716224110226,   delta=tol)
        self.assertAlmostEqual(s_h, 0.05078620072307027, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

    def test_zero_inflated_poisson(self):
        mle = maxlike.ZeroInflatedPoisson(s=2)
        mle.model = Sum(3)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 1, 1)
        mle.model.add(Vector(np.arange(2) - .5), 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))

        g = pd.read_csv(os.path.join(data_folder, "data1.csv"), index_col=[0, 1, 2])['g']
        kwargs, _ = prepare_series(g, {
            'N': np.size, 'X': np.sum, 'Z': lambda x: (x == 0).sum()})

        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean()) / 2
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean().map(np.log)
        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_zero_inflated_poisson.csv"))
        self.assertAlmostEqual(h,   0.22075127261375144, delta=tol)
        self.assertAlmostEqual(s_h, 0.04581653463450008, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

        aic = mle.akaine_information_criterion(1)
        bic = mle.bayesian_information_criterion(1)
        self.assertAlmostEqual(aic, 3.488466590286007, delta=tol)
        self.assertAlmostEqual(bic, 3.657250374928333, delta=tol)

    def test_poisson_reg(self):
        mle = maxlike.Poisson()
        mle.model = Sum(3)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 1, 1)
        mle.model.add(Vector(np.arange(2) - .5), 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))
        mle.add_regularization([0, 1], Quadratic(0, 1))

        g = pd.read_csv(os.path.join(data_folder, "data1.csv"), index_col=[0, 1, 2])['g']
        kwargs, _ = prepare_series(g, {'N': np.size, 'X': np.sum})

        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean()) / 2
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean().map(np.log)

        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_poisson_reg.csv"))
        self.assertAlmostEqual(h,   0.2754693450042746, delta=tol)
        self.assertAlmostEqual(s_h, 0.051173934224090695, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

    def test_poisson_sum(self):
        mle = maxlike.Poisson()
        mle.model = Sum(3)

        f1 = Sum(2)
        f1.add(X(), 0, 0)
        f1.add(-X(), 1, 1)
        f1.add(Scalar(), 2, None)

        f2 = Sum(2)
        f2.add(X(), 0, 0)
        f2.add(-X(), 1, 1)
        f2.add(-Scalar(), 2, None)  

        F = Sum(2)
        F.add(Exp() @ f1, [0, 1, 2], [0, 1])
        F.add(Exp() @ f2, [0, 1, 2], [1, 0])

        mle.model = Log() @ F
        mle.add_constraint([0, 1], Linear([1, 1]))

        df = pd.read_csv(os.path.join(data_folder, "data2.csv"), index_col=[0, 1])
        kwargs, _ = prepare_series(df.sum(1), {'N': np.size, 'X': np.sum})

        h = np.log(df.g1.mean()) - np.log(df.g2.mean())
        log_mean = np.log(df.mean().mean()) / 2
        a = df.groupby('t1').g1.mean() + df.groupby('t2').g2.mean()
        b = df.groupby('t1').g2.mean() + df.groupby('t2').g1.mean()
        a = np.log(a / 2) - log_mean
        b = log_mean - np.log(b / 2)

        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_poisson_sum.csv"))
        self.assertAlmostEqual(h,   0.2664172829846478, delta=tol)
        self.assertAlmostEqual(s_h, 0.18420797658168903, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

    def test_logistic(self):
        mle = maxlike.Logistic()
        mle.model = Sum(3)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 1, 1)
        mle.model.add(Vector(np.arange(2) - .5), 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))
        g = pd.read_csv(os.path.join(data_folder, "data1.csv"), index_col=[0, 1, 2])['g'] > 0
        kwargs, _ = prepare_series(g, {'N': np.size, 'X': np.sum})
        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean())
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean().map(np.log)

        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_logistic.csv"))
        self.assertAlmostEqual(h,   0.4060378612453205, delta=tol)
        self.assertAlmostEqual(s_h, 0.1321279480761052, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

    def test_negative_binomial(self):
        mle = maxlike.NegativeBinomial()
        mle.model = Sum(3)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 1, 1)
        mle.model.add(Vector(np.arange(2) - .5), 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))
        g = pd.read_csv(os.path.join(data_folder, "data1.csv"), index_col=[0, 1, 2])['g']
        kwargs, _ = prepare_series(g, {'N': np.size, 'X': np.sum})
        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean()) / 2
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean().map(np.log)

        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)
        tol = 1e-8
        mle.fit(tol=tol, **kwargs)
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_negative_binomial.csv"))
        self.assertAlmostEqual(h,   0.25833036122242375, delta=tol)
        self.assertAlmostEqual(s_h, 0.07857084780413058, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

    def test_finite(self):
        n = 8
        mle = maxlike.Finite()
        foo = Sum(3)
        foo.add(X(), 0, 0)
        foo.add(-X(), 1, 1)
        foo.add(Vector(np.arange(2) - .5), 2, 2)
        mle.model = Poisson(n) @ Exp() @ foo
        mle.add_constraint([0, 1], Linear([1, 1]))
        g = pd.read_csv(os.path.join(data_folder, "data1.csv"), index_col=[0, 1, 2])['g']
        kwargs, _ = prepare_series(df_count(g, n).stack(), {'N': np.sum})
        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean())
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean().map(np.log)

        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(**kwargs, tol=tol, verbose=self.verbose)
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_finite.csv"))
        self.assertAlmostEqual(h,   0.2805532986558426, delta=tol)
        self.assertAlmostEqual(s_h, 0.0514227784627934, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

    def test_finite_negative_binomial(self):
        n = 8
        mle = maxlike.Finite()
        foo = Sum(3)
        foo.add(X(), 0, 0)
        foo.add(-X(), 1, 1)
        foo.add(Vector(np.arange(2) - .5), 2, 2)
        mle.model = NegativeBinomial(n, 1) @ Exp() @ foo
        mle.add_constraint([0, 1], Linear([1, 1]))
        g = pd.read_csv(os.path.join(data_folder, "data1.csv"), index_col=[0, 1, 2])['g']
        kwargs, _ = prepare_series(df_count(g, n).stack(), {'N': np.sum})
        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean())
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean().map(np.log)

        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(**kwargs, tol=tol, verbose=self.verbose)
        a, b, h = mle.get_params()
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_finite_negative_binomial.csv"))
        self.assertAlmostEqual(h,   0.32819348893641703, delta=tol)
        self.assertAlmostEqual(s_h, 0.09379697910018236, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)


if __name__ == '__main__':
    unittest.main(verbosity=1)
