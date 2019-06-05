import os
import unittest
import pandas as pd
import numpy as np
import maxlike
from scipy.special import logit
from maxlike.skellam import skellam_cdf_root
from maxlike.preprocessing import prepare_dataframe, prepare_series, df_count
from maxlike.func import (
    X, Vector, Linear, Quadratic, Exp, Constant, Scalar, 
    Poisson, NegativeBinomial, Sum, Product, CollapseMatrix, MarkovMatrix)

np.seterr(all='raise', under='ignore')
maxlike.tensor.set_dtype(np.float32)
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
        b = log_mean - g.groupby(level='t2').mean()
        mle.add_param(a.values)
        mle.add_param(b.values)
        mle.add_param(h)

        tol=1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_poisson.csv"))
        self.assertAlmostEqual(h,   0.2749716224110226, delta=tol)
        self.assertAlmostEqual(s_h, 0.051132548678527,  delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

        aic = mle.akaine_information_criterion()
        bic = mle.bayesian_information_criterion()
        self.assertAlmostEqual(aic, 3.426092977152975, delta=tol)
        self.assertAlmostEqual(bic, 3.5908172972062053, delta=tol)

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
        b = log_mean - g.groupby(level='t2').mean()
        a_fix = 2
        b_fix = 3
        a[a_fix] = 1
        b[b_fix] = -1
        mle.add_param(a.values, np.arange(a.size) == a_fix)
        mle.add_param(b.values, np.arange(b.size) == b_fix)
        mle.add_param(h, False)

        tol=1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_poisson1.csv"))
        self.assertAlmostEqual(h,   0.2541710859203631,  delta=tol)
        self.assertAlmostEqual(s_h, 0.04908858811901998, delta=tol)
        np.testing.assert_allclose(a[~a.mask], df.loc[~a.mask, 'a'], atol=tol)
        np.testing.assert_allclose(b[~b.mask], df.loc[~b.mask, 'b'], atol=tol)
        np.testing.assert_allclose(s_a[~a.mask], df.loc[~a.mask, 's_a'], atol=tol)
        np.testing.assert_allclose(s_b[~b.mask], df.loc[~b.mask, 's_b'], atol=tol)
        np.testing.assert_allclose(r[~(a.mask|b.mask)],
                                   df.loc[~(a.mask|b.mask), 'r_ab'], atol=tol)

        aic = mle.akaine_information_criterion()
        bic = mle.bayesian_information_criterion()
        self.assertAlmostEqual(aic, 3.6169962347811286, delta=tol)
        self.assertAlmostEqual(bic, 3.7735927665934845, delta=tol)

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
        b = log_mean - g.groupby(level='t2').mean()
        a_fix = 2
        b_fix = 3
        a[a_fix] = 1
        b[b_fix] = -1
        mle.add_param(a.values, np.arange(a.size) == a_fix)
        mle.add_param(b.values, np.arange(b.size) == b_fix)
        mle.add_param(h, False)

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()

        df = pd.read_csv(os.path.join(data_folder, "test_poisson1.csv"))
        self.assertAlmostEqual(h,   0.2541710859203631,  delta=tol)
        self.assertAlmostEqual(s_h, 0.04908858811901998, delta=tol)
        np.testing.assert_allclose(a[~a.mask], df.loc[~a.mask, 'a'], atol=tol)
        np.testing.assert_allclose(b[~b.mask], df.loc[~b.mask, 'b'],     atol=tol)
        np.testing.assert_allclose(s_a[~a.mask], df.loc[~a.mask, 's_a'], atol=tol)
        np.testing.assert_allclose(s_b[~b.mask], df.loc[~b.mask, 's_b'], atol=tol)

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
        b = log_mean - g.groupby(level='t2').mean()

        a_fix = 2
        b_fix = 3
        a[a_fix] = 1
        b[b_fix] = -1

        mle.add_param(a.values)
        mle.add_param(b.values)
        mle.add_param(h)
        mle.add_param(h1)

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h, h1 = mle.params
        s_a, s_b, s_h, s_h1 = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_poisson3.csv"))
        self.assertAlmostEqual(h,    0.23613272896129883, delta=tol)
        self.assertAlmostEqual(s_h,  0.05505120713100134, delta=tol)
        self.assertAlmostEqual(h1,   0.13188676189444215, delta=tol)
        self.assertAlmostEqual(s_h1, 0.07186638797269668, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        #np.testing.assert_allclose(r, df['r_ab'], atol=tol)

        aic = mle.akaine_information_criterion()
        bic = mle.bayesian_information_criterion()
        self.assertAlmostEqual(aic, 3.4249495039938265, delta=tol)
        self.assertAlmostEqual(bic, 3.5937332896625294, delta=tol)

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
        b = log_mean - g.groupby(level='t2').mean()
        mle.add_param(a.values)
        mle.add_param(b.values)
        mle.add_param(h)

        tol=1e-8
        mle.fit(**kwargs, method="broyden", verbose=self.verbose)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_poisson_broyden.csv"))
        self.assertAlmostEqual(h,   0.2749716224110226,   delta=tol)
        self.assertAlmostEqual(s_h, 0.029664036960567266, delta=tol)
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
        b = log_mean - g.groupby(level='t2').mean()
        mle.add_param(a.values)
        mle.add_param(b.values)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_zero_inflated_poisson.csv"))
        self.assertAlmostEqual(h,   0.22075127261375144, delta=tol)
        self.assertAlmostEqual(s_h, 0.0458165945789084,  delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)

        aic = mle.akaine_information_criterion(1)
        bic = mle.bayesian_information_criterion(1)
        self.assertAlmostEqual(aic, 3.488466836070853, delta=tol)
        self.assertAlmostEqual(bic, 3.6572506217395557, delta=tol)

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
        b = log_mean - g.groupby(level='t2').mean()

        mle.add_param(a.values)
        mle.add_param(b.values)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()

        df = pd.read_csv(os.path.join(data_folder, "test_poisson_reg.csv"))
        self.assertAlmostEqual(h,   0.2754693450042746, delta=tol)
        self.assertAlmostEqual(s_h, 0.0511739425670764, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)

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
        a = np.log(g.groupby(level='t1').mean()) - log_mean
        b = log_mean - np.log(g.groupby(level='t2').mean())
        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)
        tol = 1e-8
        mle.fit(**kwargs, verbose=self.verbose)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()

        df = pd.read_csv(os.path.join(data_folder, "test_logistic.csv"))
        self.assertAlmostEqual(h,   0.4060377771971094, delta=tol)
        self.assertAlmostEqual(s_h, 0.1321279480761052, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)

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
        b = log_mean - g.groupby(level='t2').mean()
        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)
        tol = 1e-8
        mle.fit(tol=tol, **kwargs)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()

        df = pd.read_csv(os.path.join(data_folder, "test_negative_binomial.csv"))
        self.assertAlmostEqual(h,   0.25833036122242375, delta=tol)
        self.assertAlmostEqual(s_h, 0.07857005820984087, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)

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
        a = np.log(g.groupby(level='t1').mean()) - log_mean
        b = log_mean - np.log(g.groupby(level='t2').mean())
        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(**kwargs, tol=tol, verbose=self.verbose)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()

        df = pd.read_csv(os.path.join(data_folder, "test_finite.csv"))
        self.assertAlmostEqual(h,   0.2805532986558426, delta=tol)
        self.assertAlmostEqual(s_h, 0.0514227784627934, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)

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
        a = np.log(g.groupby(level='t1').mean()) - log_mean
        b = log_mean - np.log(g.groupby(level='t2').mean())
        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(**kwargs, tol=tol, verbose=self.verbose)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()
        r = np.diag(mle.error_matrix()[0][1]) / s_a / s_b

        df = pd.read_csv(os.path.join(data_folder, "test_finite_negative_binomial.csv"))
        self.assertAlmostEqual(h,   0.32819346226217844, delta=tol)
        self.assertAlmostEqual(s_h, 0.09379697849393093, delta=tol)
        np.testing.assert_allclose(a, df['a'], atol=tol)
        np.testing.assert_allclose(b, df['b'], atol=tol)
        np.testing.assert_allclose(s_a, df['s_a'], atol=tol)
        np.testing.assert_allclose(s_b, df['s_b'], atol=tol)
        np.testing.assert_allclose(r, df['r_ab'], atol=tol)


if __name__ == '__main__':
    unittest.main(verbosity=1)
