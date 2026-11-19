import io
import unittest
import contextlib
import numpy as np
import maxlike
from maxlike.func import X, Vector, Sum, Linear


def poisson_model(n=6, constraint=True, seed=0):
    """
    Small attack/defence Poisson model on synthetic data:
        log E[x_{ij,k}] = a_i - b_j + h * (k - 1/2)
    """
    rng = np.random.RandomState(seed)
    a = rng.normal(0, .3, n)
    b = rng.normal(0, .3, n)
    v = np.arange(2) - .5
    lam = np.exp(a[:, None, None] - b[None, :, None] + .25 * v)
    N = np.full((n, n, 2), 20.)
    X_obs = rng.poisson(lam * N).astype(float)

    mle = maxlike.Poisson()
    mle.model = Sum(3)
    mle.model.add(X(), 0, 0)
    mle.model.add(-X(), 1, 1)
    mle.model.add(Vector(v), 2, 2)
    if constraint:
        mle.add_constraint([0, 1], Linear([1, 1]))
        mle.add_param(np.zeros(n))
    else:
        # fix a_0 = 0 to remove the a/b translation invariance
        mle.add_param(np.zeros(n), np.arange(n) == 0)
    mle.add_param(np.zeros(n))
    mle.add_param(0.)
    return mle, {'N': N, 'X': X_obs}


class Test(unittest.TestCase):

    def test_broyden_verbose(self):
        mle, kwargs = poisson_model()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertTrue(mle.fit(method="broyden", verbose=1,
                                    max_steps=50, **kwargs))
        self.assertGreater(len(out.getvalue().splitlines()), 1)

    def test_std_error_without_constraints(self):
        mle, kwargs = poisson_model(constraint=False)
        mle.fit(**kwargs)
        s_a, s_b, s_h = mle.std_error()
        self.assertTrue(s_a.mask[0])  # fixed parameter has no std error
        self.assertTrue(np.all(np.isfinite(s_a.compressed())))
        self.assertTrue(np.all(s_b > 0))
        self.assertGreater(float(s_h), 0)
        # consistent with the diagonal of the covariance matrix
        cov = mle.error_matrix()
        np.testing.assert_allclose(np.diag(cov[1][1]), s_b ** 2, rtol=1e-8)
        np.testing.assert_allclose(float(cov[2][2]), float(s_h) ** 2,
                                   rtol=1e-8)

    def test_converged_flag(self):
        mle, kwargs = poisson_model()
        self.assertFalse(mle.converged)
        self.assertTrue(mle.fit(**kwargs))
        self.assertTrue(mle.converged)
        # failing to converge resets the flag
        with self.assertRaises(maxlike.ConvergenceError):
            mle.fit(max_steps=0, **kwargs)
        self.assertFalse(mle.converged)
