import unittest
import numpy as np
import pandas as pd
from scipy.stats import norm
import maxlike
from maxlike.func import (
    X, Vector, Sum, Linear, Exp, Poisson, Logistic)
from maxlike.inference import free_covariance

TEAMS = ["T%d" % k for k in range(6)]


def football(seed=0, n_games=30.):
    rng = np.random.default_rng(seed)
    n = len(TEAMS)
    a, b = rng.normal(0, .3, (2, n))
    v = np.arange(2) - .5
    N = np.full((n, n, 2), n_games)
    np.einsum('iik->ik', N)[:] = 0
    lam = np.exp(a[:, None, None] - b[None, :, None] + .25 * v)
    X_ = rng.poisson(N * lam).astype(float)
    mle = maxlike.Poisson()
    mle.model = Sum(3)
    mle.model.add(X(), 0, 0)
    mle.model.add(-X(), 1, 1)
    mle.model.add(Vector(v), 2, 2)
    mle.add_constraint([0, 1], Linear([1, 1]))
    mle.add_param(pd.Series(np.zeros(n), index=TEAMS), name="attack")
    mle.add_param(np.zeros(n), np.arange(n) == 0, name="defence",
                  labels=TEAMS)
    mle.add_param(0., name="home")
    return mle, {"N": N, "X": X_}


class TestLabels(unittest.TestCase):

    def test_labels_and_summary(self):
        mle, data = football()
        mle.fit(**data)
        attack, defence, home = mle.get_params(labeled=True)
        self.assertEqual(list(attack.index), TEAMS)
        self.assertEqual(attack.name, "attack")
        self.assertEqual(defence["T0"], 0)  # fixed entry keeps its value
        self.assertIsInstance(home, float)

        s = mle.summary()
        self.assertEqual(list(s.index.get_level_values(0).unique()),
                         ["attack", "defence", "home"])
        self.assertEqual(len(s), 2 * len(TEAMS) + 1)
        est = np.ma.getdata(mle.get_params()[0])
        se = np.ma.getdata(mle.std_error()[0])
        np.testing.assert_allclose(s.loc["attack", "estimate"], est)
        np.testing.assert_allclose(s.loc["attack", "std_error"], se)
        q = norm.ppf(.975)
        np.testing.assert_allclose(s.loc["attack", "ci_95%_lower"],
                                   est - q * se)
        np.testing.assert_allclose(s.loc["attack", "p_value"],
                                   2 * norm.sf(np.abs(est / se)))
        # a fixed entry has no standard error
        row = s.loc[("defence", "T0")]
        self.assertTrue(row["fixed"])
        self.assertTrue(np.isnan(row["std_error"]))

    def test_unnamed_params(self):
        mle = maxlike.Poisson()
        mle.add_param(np.zeros(3))
        self.assertEqual(mle.get_params(labeled=True)[0].name, "param 0")


class TestPredict(unittest.TestCase):

    def setUp(self):
        self.mle, data = football()
        self.mle.fit(**data)

    def test_value(self):
        np.testing.assert_allclose(
            self.mle.predict(),
            np.asarray(self.mle.model(self.mle.params).toarray()))

    def test_se_of_a_parameter(self):
        # predicting a parameter itself gives its standard error
        attack = Sum(1)
        attack.add(X(), 0, 0)
        value, se = self.mle.predict(attack, se=True)
        np.testing.assert_allclose(value, np.ma.getdata(
            self.mle.get_params()[0]))
        np.testing.assert_allclose(se, np.ma.getdata(
            self.mle.std_error()[0]), rtol=1e-10)

    def test_delta_method_matches_numeric_jacobian(self):
        # expected number of goals: exp(model)
        target = Exp() @ self.mle.model
        value, se = self.mle.predict(target, se=True)

        mle = self.mle
        flat = np.concatenate([p.compressed() for p in mle.params])
        h = 1e-6
        cols = []
        for k in range(flat.size):
            e = np.zeros_like(flat)
            e[k] = h
            up = mle.predict(target, params=mle._reshape_params(flat + e))
            dn = mle.predict(target, params=mle._reshape_params(flat - e))
            cols.append(((up - dn) / (2 * h)).ravel())
        jac = np.array(cols)
        var = np.einsum("ic,ij,jc->c", jac, free_covariance(mle), jac)
        np.testing.assert_allclose(se.ravel(), np.sqrt(var), rtol=1e-5)
        self.assertEqual(value.shape, se.shape)


class TestSimulate(unittest.TestCase):

    def test_poisson(self):
        mle, data = football()
        mle.fit(**data)
        sim = mle.simulate(rng=1)
        self.assertEqual(set(sim), {"N", "X"})
        self.assertEqual(sim["X"].shape, data["X"].shape)
        np.testing.assert_array_equal(sim["X"], np.rint(sim["X"]))
        np.testing.assert_array_equal(sim["X"], mle.simulate(rng=1)["X"])
        # zero counts give zero outcomes; the mean follows the model
        self.assertEqual(sim["X"][data["N"] == 0].sum(), 0)
        mean = np.mean([mle.simulate(rng=s)["X"].sum() for s in range(200)])
        expected = (data["N"] * np.exp(mle.predict())).sum()
        self.assertLess(abs(mean / expected - 1), 0.01)

    def test_parametric_bootstrap(self):
        mle, data = football()
        mle.fit(**data)
        boot, _ = football()
        self.assertTrue(boot.fit(**mle.simulate(rng=3)))

    def test_logistic(self):
        rng = np.random.default_rng(0)
        n = 5
        F = Sum(2)
        F.add(X(), 0, 0)
        F.add(-X(), 0, 1)
        N = np.full((n, n), 10.)
        mle = maxlike.Logistic()
        mle.model = F
        mle.add_constraint(0, Linear([1]))
        mle.add_param(rng.normal(0, .3, n))
        mle.N = maxlike.Tensor(N)
        sim = mle.simulate(rng=0)
        self.assertTrue(np.all((sim["X"] >= 0) & (sim["X"] <= N)))

    def test_finite_keeps_row_totals(self):
        rng = np.random.default_rng(0)
        f = Sum(1)
        f.add(X(), 0, 0)
        mle = maxlike.Finite()
        mle.model = Poisson(4) @ Exp() @ f
        mle.add_param(rng.normal(0, .3, 5))
        N = rng.integers(0, 5, (5, 4)).astype(float)
        mle.N = maxlike.Tensor(N, dim=1)
        sim = mle.simulate(rng=0)
        np.testing.assert_array_equal(sim["N"].sum(-1), N.sum(-1))


if __name__ == "__main__":
    unittest.main()
