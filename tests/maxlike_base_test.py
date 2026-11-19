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
