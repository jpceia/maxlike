import unittest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "..")
import maxlike
from maxlike.func import (
    X, Vector, Linear, Quadratic, Compose, Exp, Constant, Scalar, 
    Poisson, Sum, Product, CollapseMatrix)


class Test(unittest.TestCase):

    def test_poisson(self):
        mle = maxlike.Poisson()
        mle.model = Sum(3)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 1, 1)
        mle.model.add(Vector(np.arange(2) - .5), 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))

        g = pd.read_csv(r"data\data1.csv", index_col=[0, 1, 2])['g']
        prepared_data, _ = maxlike.utils.prepare_series(
            g, {'N': np.size, 'X': np.sum})

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
        mle.fit(tol=tol, **prepared_data)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()

        self.assertAlmostEqual(h,   0.2541711117084739,  delta=tol)
        self.assertAlmostEqual(s_h, 0.04908832460966404, delta=tol)

        df = pd.read_csv(r"data\test_poisson.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

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

        g = pd.read_csv(r"data\data1.csv", index_col=[0, 1, 2])['g']
        prepared_data, _ = maxlike.utils.prepare_series(
            g, {'N': np.size, 'X': np.sum})

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
        mle.fit(tol=tol, **prepared_data)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()
        
        self.assertAlmostEqual(h,   0.2541711117084739,  delta=tol)
        self.assertAlmostEqual(s_h, 0.04908832460966404, delta=tol)

        df = pd.read_csv(r"data\test_poisson.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

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

        g = pd.read_csv(r"data\data1.csv", index_col=[0, 1, 2])['g']
        prepared_data, _ = maxlike.utils.prepare_series(
            g, {'N': np.size, 'X': np.sum})

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
        mle.fit(tol=tol, **prepared_data)
        a, b, h, h1 = mle.params
        s_a, s_b, s_h, s_h1 = mle.std_error()

        self.assertAlmostEqual(h,    0.23613271940658626, delta=tol)
        self.assertAlmostEqual(s_h,  0.05505120716062757, delta=tol)
        self.assertAlmostEqual(h1,   0.13188677053914621, delta=tol)
        self.assertAlmostEqual(s_h1, 0.07186639159291758, delta=tol)

        df = pd.read_csv(r"data\test_poisson3.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

    def test_poisson_reg(self):
        mle = maxlike.Poisson()
        mle.model = Sum(3)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 1, 1)
        mle.model.add(Vector(np.arange(2) - .5), 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))
        mle.add_regularization([0, 1], Quadratic(0, 1))

        g = pd.read_csv(r"data\data1.csv", index_col=[0, 1, 2])['g']
        prepared_data, _ = maxlike.utils.prepare_series(
            g, {'N': np.size, 'X': np.sum})

        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean()) / 2
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean()

        mle.add_param(a.values)
        mle.add_param(b.values)
        mle.add_param(h)

        tol = 1e-8
        mle.fit(tol=tol, **prepared_data)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()

        self.assertAlmostEqual(h,   0.2754693439235266, delta=tol)
        self.assertAlmostEqual(s_h, 0.05117380388073695, delta=tol)
        df = pd.read_csv(r"data\test_poisson_reg.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

    def test_logistic(self):
        mle = maxlike.Logistic()
        mle.model = Sum(3)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 1, 1)
        mle.model.add(Vector(np.arange(2) - .5), 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))
        g = pd.read_csv(r"data\data1.csv", index_col=[0, 1, 2])['g'] > 0
        prepared_data, _ = maxlike.utils.prepare_series(
            g, {'N': np.size, 'X': np.sum})
        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean())
        a = np.log(g.groupby(level='t1').mean()) - log_mean
        b = log_mean - np.log(g.groupby(level='t2').mean())
        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)
        tol = 1e-8
        mle.fit(tol=tol, **prepared_data)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()

        self.assertAlmostEqual(h,   0.4060378612453205, delta=tol)
        self.assertAlmostEqual(s_h, 0.13212794903767752, delta=tol)
        df = pd.read_csv(r"data\test_logistic.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

    def test_logistic_cross(self):
        mle = maxlike.Logistic()
        mle.model = Sum(2)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 0, 1)
        mle.model.add(Scalar(), 1, [])
        mle.add_constraint([0], Linear([1]))

        # fetch and prepare data
        df = pd.read_csv(r"data\test_data_proba.csv", index_col=[0, 1, 2])['p'].unstack(level=2)
        _N = maxlike.utils.prepare_series(df[-1] + df[1], {'N': np.sum})[0]['N']
        _X = maxlike.utils.prepare_series(df[1], {'X': np.sum})[0]['X']
        prepared_data = {'N': _N, 'X': _X}
        a = 4 * _X.sum(0) / _N.sum(0)
        a -= a.mean()
        
        mle.add_param(a)
        mle.add_param(0)

        tol = 1e-8
        mle.fit(tol=tol, **prepared_data)
        a, h = mle.params
        s_a, s_h = mle.std_error()

    def test_negative_binomial(self):
        mle = maxlike.NegativeBinomial()
        mle.model = Sum(3)
        mle.model.add(X(), 0, 0)
        mle.model.add(-X(), 1, 1)
        mle.model.add(Vector(np.arange(2) - .5), 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))
        g = pd.read_csv(r"data\data1.csv", index_col=[0, 1, 2])['g']
        prepared_data, _ = maxlike.utils.prepare_series(
            g, {'N': np.size, 'X': np.sum})
        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean())
        a = np.log(g.groupby(level='t1').mean()) - log_mean
        b = log_mean - np.log(g.groupby(level='t2').mean())
        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)
        tol = 1e-8
        mle.fit(tol=tol, **prepared_data)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()

        self.assertAlmostEqual(h,   0.2583303652037732, delta=tol)
        self.assertAlmostEqual(s_h, 0.07857076508033528, delta=tol)
        df = pd.read_csv(r"data\test_negative_binomial.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

    def test_finite(self):
        n = 8
        mle = maxlike.Finite()
        foo = Sum(3)
        foo.add(X(), 0, 0)
        foo.add(-X(), 1, 1)
        foo.add(Vector(np.arange(2) - .5), 2, 2)
        mle.model = Compose(Poisson(n), Compose(Exp(), foo))
        mle.add_constraint([0, 1], Linear([1, 1]))
        g = pd.read_csv(r"data\data1.csv", index_col=[0, 1, 2])['g']
        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean())
        a = np.log(g.groupby(level='t1').mean()) - log_mean
        b = log_mean - np.log(g.groupby(level='t2').mean())
        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)
        tol = 1e-8
        prepared_data, _ = maxlike.utils.prepare_series(
            maxlike.utils.df_count(g, n).stack(), {'N': np.sum})
        mle.fit(tol=tol, **prepared_data)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()

        self.assertAlmostEqual(h,   0.28055330673303397, delta=tol)
        self.assertAlmostEqual(s_h, 0.05142277849813604, delta=tol)
        df = pd.read_csv(r"data\test_finite.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

    def test_finite_matrix(self): 
        n = 8

        mle = maxlike.Finite(dim=2)

        # guess params
        g = pd.read_csv(r"data\data1.csv", index_col=[0, 1, 2])['g']
        h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
        log_mean = np.log(g.mean())
        a = np.log(g.groupby(level='t1').mean()) - log_mean
        b = log_mean - np.log(g.groupby(level='t2').mean())

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

        # fetch and prepare data
        df = pd.read_csv(r"data\data_finite_matrix.csv",
                         index_col=[0, 1], header=[0, 1]).stack([0, 1])
        prepared_data, _ = maxlike.utils.prepare_series(df)

        # calibration
        tol = 1e-8
        mle.fit(tol=tol, **prepared_data)
        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()

        self.assertAlmostEqual(h,   0.2785288251171003, delta=tol)
        self.assertAlmostEqual(s_h, 0.05147212068942573, delta=tol)
        df = pd.read_csv(r"data\test_finite_matrix.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

    @unittest.skip
    def test_kullback_leibler(self):
        n = 8
        mle = maxlike.Finite()

        # fetch and prepare data
        df = pd.read_csv(r"data\test_data_proba.csv", index_col=[0, 1, 2])['p']
        prepared_data, _ = maxlike.utils.prepare_series(df, {'N': np.sum})

        # guess params
        a = np.zeros((18))
        b = np.zeros((18))
        h = 0

        mle.add_param(a)
        mle.add_param(b)
        mle.add_param(h)
        mle.add_constraint([0, 1], Linear([1, 1]))

        # define functions
        L = 10
        f1 = Sum(2)
        f1.add(X(), 0, 0)
        f1.add(-X(), 1, 1)
        f1.add(0.5 * Scalar(), 2, [])

        f2 = Sum(2)
        f2.add(X(), 0, 0)
        f2.add(-X(), 1, 1)
        f2.add(-0.5 * Scalar(), 2, [])

        F1 = Poisson(L) @ Exp() @ f1
        F2 = Poisson(L) @ Exp() @ f2

        G = Product(2, 2)
        G.add(F1, [0, 1, 2], [0, 1], 0)
        G.add(F2, [1, 0, 2], [0, 1], 1)

        mle.model = CollapseMatrix() @ G

        mle.fit(verbose=2, max_steps=50, **prepared_data)

        a, b, h = mle.params
        s_a, s_b, s_h = mle.std_error()

        df = pd.DataFrame({'a': a, 'b': b})

if __name__ == '__main__':
    unittest.main()
