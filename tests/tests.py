import unittest
import pandas as pd
import numpy as np
# import sys
# sys.path.insert(0, "..")
import maxlike
from maxlike.func import Encode, Vector, Linear


class Test(unittest.TestCase):

    def test_poisson(self):
        mle = maxlike.Poisson()
        mle.verbose = False
        mle.model = maxlike.func.Sum(3)
        mle.model.add(0, 0, Encode())
        mle.model.add(1, 1, -Encode())
        mle.add_constraint([0, 1], Linear([1, 1]))
        mle.model.add(2, 2, Vector(np.arange(2) - .5))
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
        self.assertAlmostEqual(s_h, 0.0804417430336552, delta=tol)
        df = pd.read_csv("test_results1.csv")
        self.assertTrue(np.allclose(a, df['a'].values, atol=tol))
        self.assertTrue(np.allclose(b, df['b'].values, atol=tol))
        self.assertTrue(np.allclose(s_a, df['s_a'].values, atol=tol))
        self.assertTrue(np.allclose(s_b, df['s_b'].values, atol=tol))


if __name__ == '__main__':
    unittest.main()
