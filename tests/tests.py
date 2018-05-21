import unittest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "..")
import maxlike
from maxlike.func import (
    Encode, Vector, Linear, Quadratic, Compose, Exp, Constant, Scalar,
    Poisson, Sum, Product, CollapseMatrix)


class Test(unittest.TestCase):

    def test_poisson(self):
        mle = maxlike.Poisson()
        mle.model = maxlike.func.Sum(3)
        mle.model.add(Encode(), 0, 0)
        mle.model.add(-Encode(), 1, 1)
        mle.model.add(Vector(np.arange(2) - .5), 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))
        g = pd.read_csv("test_data1.csv", index_col=[0, 1, 2])['g']
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
        a, b, h = mle.params_
        s_a, s_b, s_h = mle.std_error()
        self.assertAlmostEqual(h.data, 0.3496149212379256, delta=tol)
        self.assertAlmostEqual(s_h, 0.0804417430337, delta=tol)
        df = pd.read_csv("test_results1.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

    def test_poisson2(self):
        mle = maxlike.Poisson()
        mle.model = Sum(3)
        s = Sum(2)
        s.add(Encode(), 0, 0)
        s.add(-Encode(), 1, 1)
        k = Constant(np.arange(2) - .5)
        f_h = Product(1).add(k, None, 0).add(Scalar(), 0, None)
        mle.model.add(s, [0, 1], [0, 1])
        mle.model.add(f_h, 2, 2)
        mle.add_constraint([0, 1], Linear([1, 1]))
        g = pd.read_csv("test_data1.csv", index_col=[0, 1, 2])['g']
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
        a, b, h = mle.params_
        s_a, s_b, s_h = mle.std_error()
        self.assertAlmostEqual(h.data, 0.3496149212379256, delta=tol)
        self.assertAlmostEqual(s_h, 0.0804417430337, delta=tol)
        df = pd.read_csv("test_results1.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

    def test_poisson3(self):
        mle = maxlike.Poisson()
        mle.model = Sum(3)
        s = Sum(2).add(Encode(), 0, 0).add(-Encode(), 1, 1)
        prod = Product(2).add(s, [0, 1], [0, 1]).add(Scalar(), 2, None)
        s_h = Sum(2).add(Scalar(), 2, None).add(prod, [0, 1, 3], [0, 1])
        k = Constant(np.arange(2) - .5)
        f_h = Product(3).add(k, None, 2).add(s_h, [0, 1, 2, 3], [0, 1])
        mle.model.add(s, [0, 1], [0, 1])
        mle.model.add(f_h, [0, 1, 2, 3], [0, 1, 2])
        mle.add_constraint([0, 1], Linear([1, 1]))
        g = pd.read_csv("test_data1.csv", index_col=[0, 1, 2])['g']
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
        mle.add_param(a.values, np.arange(a.size) == a_fix)
        mle.add_param(b.values, np.arange(b.size) == b_fix)
        mle.add_param(h, False)
        mle.add_param(h1, False)
        tol = 1e-8
        mle.fit(tol=tol, verbose=True, **prepared_data)
        a, b, h = mle.params_
        s_a, s_b, s_h = mle.std_error()

    def test_poisson_reg(self):
        mle = maxlike.Poisson()
        mle.model = Sum(2)
        mle.model.add(Encode(), 0, 0)
        mle.model.add(-Encode(), 1, 1)
        mle.add_constraint([0, 1], Linear([1, 1]))
        mle.add_regularization([0, 1], Quadratic(0, 1))
        g = pd.read_csv("test_data1.csv", index_col=[0, 1])['g']
        prepared_data, _ = maxlike.utils.prepare_series(
            g, {'N': np.size, 'X': np.sum})
        log_mean = np.log(g.mean()) / 2
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean()
        mle.add_param(a)
        mle.add_param(b)
        tol = 1e-8
        mle.fit(tol=tol, **prepared_data)
        a, b = mle.params_
        s_a, s_b = mle.std_error()
        df = pd.read_csv("test_results_reg.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

    def test_logistic(self):
        mle = maxlike.Logistic()
        mle.model = Sum(2)
        mle.model.add(Encode(), 0, 0)
        mle.model.add(-Encode(), 1, 1)
        mle.add_constraint([0, 1], Linear([1, 1]))
        g = pd.read_csv("test_data1.csv", index_col=[0, 1])['g'] > 1
        prepared_data, _ = maxlike.utils.prepare_series(
            g, {'N': np.size, 'X': np.sum})
        m = np.log(g.mean())
        a = np.log(g.groupby(level='t1').mean()) - m
        b = m - np.log(g.groupby(level='t2').mean())
        mle.add_param(a)
        mle.add_param(b)
        tol = 1e-8
        mle.fit(tol=tol, **prepared_data)
        a, b = mle.params_
        s_a, s_b = mle.std_error()
        df = pd.read_csv("test_results_logistic.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

    def test_negative_binomial(self):
        mle = maxlike.NegativeBinomial()
        mle.model = Sum(2)
        mle.model.add(Encode(), 0, 0)
        mle.model.add(-Encode(), 1, 1)
        mle.add_constraint([0, 1], Linear([1, 1]))
        g = pd.read_csv("test_data1.csv", index_col=[0, 1])['g']
        prepared_data, _ = maxlike.utils.prepare_series(
            g, {'N': np.size, 'X': np.sum})
        log_mean = np.log(g.mean()) / 2
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean()
        mle.add_param(a)
        mle.add_param(b)
        tol = 1e-8
        mle.fit(tol=tol, **prepared_data)
        a, b = mle.params_
        s_a, s_b = mle.std_error()
        df = pd.read_csv("test_negative_binomial.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

    def test_finite(self):
        L = 10
        mle = maxlike.Finite()
        foo = Sum(2)
        foo.add(Encode(), 0, 0)
        foo.add(-Encode(), 1, 1)
        mle.model = Compose(Poisson(L), Compose(Exp(), foo))
        mle.add_constraint([0, 1], Linear([1, 1]))
        g = pd.read_csv("test_data1.csv", index_col=[0, 1])['g']
        log_mean = np.log(g.mean()) / 2
        a = g.groupby(level='t1').mean().map(np.log) - log_mean
        b = log_mean - g.groupby(level='t2').mean()
        mle.add_param(a)
        mle.add_param(b)
        tol = 1e-8
        prepared_data, _ = maxlike.utils.prepare_series(
            maxlike.utils.df_count(g, L).stack(), {'N': np.sum})
        mle.fit(tol=tol, **prepared_data)
        a, b = mle.params_
        s_a, s_b = mle.std_error()
        df = pd.read_csv("test_finite.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))

    def test_kullback_leibler(self):
        
        mle = maxlike.Finite()

        # fetch and prepare data
        df = pd.read_csv("test_data_proba.csv", index_col=[0, 1, 2])['p']
        prepared_data, _ = maxlike.utils.prepare_series(df, {'N': np.sum})

        # guess params
        a = np.zeros((18))
        b = np.zeros((18))
        #h = .2

        mle.add_param(a)
        mle.add_param(b)
        #mle.add_param(h)
        mle.add_constraint([0, 1], Linear([1, 1]))

        # define functions
        L = 10
        f = maxlike.func.Sum(2)
        f.add(Encode(), 0, 0)
        f.add(-Encode(), 1, 1)
        foo = Poisson(L) @ Exp() @ f

        F = Product(2, 2)
        F.add(foo, [0, 1], [0, 1], 0)
        F.add(foo, [1, 0], [0, 1], 1)

        mle.model = CollapseMatrix() @ F
        tol = 1e-8

        mle.fit(tol=tol, max_steps=20, **prepared_data)
        a, b = mle.params_
        s_a, s_b = mle.std_error()


if __name__ == '__main__':
    unittest.main()
