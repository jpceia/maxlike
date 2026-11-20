"""
Runs every script in docs/examples so that they keep working, and checks a
few properties of their results.
"""
import io
import os
import sys
import contextlib
import unittest
import numpy as np

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "docs", "examples")
sys.path.insert(0, EXAMPLES)

import football_goals  # noqa: E402
import football_scores  # noqa: E402
import football_market_calibration  # noqa: E402
import football_dynamic  # noqa: E402
import tennis  # noqa: E402
from maxlike.func import CollapseMatrix, Collapse  # noqa: E402


def run(module):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = module.main()
    return result, out.getvalue()


class Test(unittest.TestCase):

    def test_football_goals(self):
        mle, out = run(football_goals)
        self.assertIn("Home advantage", out)
        home = float(mle.get_params()[2])
        self.assertAlmostEqual(home, 0.2749716, delta=1e-4)

    def test_football_scores(self):
        mle, out = run(football_scores)
        # close to tests/matrix_test.py::test_poisson_matrix, which uses the
        # truncated (unnormalised) Poisson instead of the censored one
        self.assertAlmostEqual(float(mle.get_params()[2]),
                               0.27852882496320425, delta=5e-3)
        probs = mle.predict(CollapseMatrix() @ mle.model)
        np.testing.assert_allclose(probs.sum(-1), 1, atol=1e-12)

    def test_football_market_calibration(self):
        mle, out = run(football_market_calibration)
        a, d, h = [np.ma.getdata(p) for p in mle.get_params()]
        # plausible scoring rates (no degenerate fit at the grid cap)
        self.assertLess(np.exp(a[:, None] - d[None, :] + h / 2).max(), 5)

    def test_football_dynamic(self):
        (dynamic, static, truth), out = run(football_dynamic)
        self.assertIn("chosen sigma: 0.050", out)
        # the dynamic fit follows the true strengths better than the static
        true_a = truth["attack"].values
        a_static = np.ma.getdata(static.get_params()[0])[:, None]
        a_dynamic = np.ma.getdata(dynamic.get_params()[0])
        self.assertLess(np.mean((a_dynamic - true_a) ** 2),
                        .8 * np.mean((a_static - true_a) ** 2))
        self.assertAlmostEqual(float(dynamic.get_params()[2]),
                               truth["home"], delta=.1)

    def test_tennis(self):
        mle, out = run(tennis)
        best = mle.summary().loc["strength", "estimate"].idxmax()
        self.assertEqual(best, "Nadal R.")
        self.assertIn("Nadal R. v Djokovic N.", out)
        p = mle.predict(Collapse(tennis.model(), False, True))
        # P(i wins a set against j) + P(j wins it against i) = 1
        np.testing.assert_allclose(p + p.T, 1, atol=1e-9)

    def test_race(self):
        import race
        with contextlib.redirect_stdout(io.StringIO()):
            runners, wt, w, e, X, truth = race.simulate(n_races=800)
            mle = race.fit(runners, wt, w, e, X, common_factor=True)
        self.assertLess(abs(abs(float(mle.get_params()[3])) - truth["tau"]),
                        4 * float(mle.std_error()[3]))


if __name__ == "__main__":
    unittest.main()
