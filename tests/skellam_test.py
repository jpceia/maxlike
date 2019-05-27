import unittest
from maxlike.skellam import (
    poisson_cdf,
    poisson_cdf_root,
    skellam_cdf,
    skellam_cdf_pair,
    skellam_cdf_root,
    conditional_skellam_cdf_root)


class Test(unittest.TestCase):

    def test_poisson_cdf_root(self):
        tol = 1e-8
        target = 0.3
        y = poisson_cdf_root(target, 2, tol=tol)
        p, d = poisson_cdf(y, 2)
        self.assertAlmostEqual(p, target, delta=tol)

    def test_skellam_cdf_root(self):
        tol = 1e-8
        target1 = 0.3
        target2 = 0.4
        a, b = skellam_cdf_root(target1, target2, 1, tol=tol)
        res = skellam_cdf_pair(a, b, 1)
        self.assertAlmostEqual(res[0, 0], target1, delta=tol)
        self.assertAlmostEqual(res[1, 0], target2, delta=tol)

    def test_skellam_triangle_root(self):
        tol = 1e-8
        target = 0.3
        y = 1.8
        x = conditional_skellam_cdf_root(y, target, tol=tol)
        t0 = skellam_cdf(x, y, -1)[0]
        t1 = skellam_cdf(x, y, 0)[0]
        t = 0.5 * (t0 + t1)
        self.assertAlmostEqual(target, t, delta=tol)


if __name__ == "__main__":
    unittest.main()
