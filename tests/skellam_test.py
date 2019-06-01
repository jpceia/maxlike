import unittest
from maxlike.skellam import (
    tol2size,
    poisson_cdf,
    poisson_mid_cdf,
    poisson_cdf_root,
    skellam_cdf,
    skellam_cdf_pair,
    skellam_cdf_root,
    conditional_skellam_cdf_root)


class Test(unittest.TestCase):

    tol = 1e-8

    def test_poisson_cdf_root(self):
        target = 0.3
        y = poisson_cdf_root(target, 2, tol=self.tol)
        p, d = poisson_cdf(y, 2)
        self.assertAlmostEqual(p, target, delta=self.tol)

    def test_skellam_cdf_root(self):
        target1 = 0.3
        target2 = 0.4
        a, b = skellam_cdf_root(target1, target2, 1, tol=self.tol)
        res = skellam_cdf_pair(a, b, 1)
        self.assertAlmostEqual(res[0, 0], target1, delta=self.tol)
        self.assertAlmostEqual(res[1, 0], target2, delta=self.tol)

    def test_skellam_triangle_root(self):
        target = 0.3
        y = 1.8
        x = conditional_skellam_cdf_root(y, target, tol=self.tol)
        t0 = skellam_cdf(x, y, -1)[0]
        t1 = skellam_cdf(x, y, 0)[0]
        t = 0.5 * (t0 + t1)
        self.assertAlmostEqual(target, t, delta=self.tol)

    def test_poisson_mid_cdf(self):
        y, d = poisson_mid_cdf(2, 2)
        self.assertAlmostEqual(y, 0.5413411329464508, delta=self.tol)
        self.assertAlmostEqual(d, -0.2706705664732254, delta=self.tol)

    def test_tol2size(self):
        self.assertEqual(tol2size(1e-12), 9)


if __name__ == "__main__":
    unittest.main()
