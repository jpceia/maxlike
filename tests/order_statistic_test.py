import os
import sys
import unittest
import numpy as np
import pandas as pd
import maxlike
from maxlike.func import FirstOrderStatistic, Linear
from .numdiff import derivative_errors

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "docs", "examples"))
from race import simulate, fit as fit_race  # noqa: E402


def small_problem(seed=0):
    rng = np.random.default_rng(seed)
    T, J = 12, 4
    entrants = rng.random((T, J)) < .6
    entrants[np.arange(T), rng.integers(0, J, T)] = True
    winner = np.array([rng.choice(np.flatnonzero(b)) for b in entrants])
    X = np.column_stack([np.ones(T), rng.normal(size=T)])
    winning_time = rng.normal(-.2, .2, T)
    return winning_time, winner, entrants, X


def fit_without_constraint(runners, winning_time, winner, entrants, X):
    mle = maxlike.LogLikelihood()
    mle.model = FirstOrderStatistic(winning_time, winner, entrants, X)
    mle.add_param(pd.Series(0., index=runners), name="mu")
    mle.add_param(np.log(.1), name="log_sigma")
    mle.add_param(np.array([winning_time.mean(), 0.]), name="beta")
    mle.fit(N=np.ones(len(winning_time)))
    return mle


class Test(unittest.TestCase):

    def test_derivatives(self):
        winning_time, winner, entrants, X = small_problem()
        rng = np.random.default_rng(1)
        mu = rng.normal(0, .1, entrants.shape[1])
        for common_factor in (False, True):
            f = FirstOrderStatistic(winning_time, winner, entrants, X,
                           common_factor=common_factor, nodes=9)
            params = [mu, np.log(.15), np.array([-.1, .05])]
            if common_factor:
                params.append(.12)
            for use_eval in (True, False):
                with self.subTest(common_factor=common_factor,
                                  use_eval=use_eval):
                    self.assertEqual(derivative_errors(
                        f, params, use_eval=use_eval), [])

    def test_independent_runners_recovered(self):
        runners, wt, w, e, X, truth = simulate(tau=0., n_races=1500)
        mle = fit_race(runners, wt, w, e, X, common_factor=False)
        mu, log_sigma, beta = mle.get_params()
        mu_se, log_sigma_se, beta_se = mle.std_error()
        self.assertLess(np.max(np.abs(mu - truth["mu"]) / mu_se), 4)
        self.assertLess(abs(float(log_sigma) - np.log(truth["sigma"])),
                        4 * float(log_sigma_se))
        np.testing.assert_array_less(np.abs(beta - truth["beta"]),
                                     4 * beta_se)

    def test_common_factor_recovered_and_needed(self):
        runners, wt, w, e, X, truth = simulate(tau=.06, n_races=2000)
        indep = fit_race(runners, wt, w, e, X, common_factor=False)
        common = fit_race(runners, wt, w, e, X, common_factor=True)

        mu, log_sigma, beta, tau = common.get_params()
        se = common.std_error()
        self.assertLess(abs(abs(float(tau)) - truth["tau"]),
                        4 * float(se[3]))
        self.assertLess(abs(float(log_sigma) - np.log(truth["sigma"])),
                        4 * float(se[1]))
        rmse = lambda m: np.sqrt(np.mean(  # noqa: E731
            (np.ma.getdata(m.get_params()[0]) - truth["mu"]) ** 2))
        # ignoring the race-day factor biases the runners' levels and the
        # spread
        self.assertLess(rmse(common), rmse(indep) / 2)
        self.assertGreater(float(np.exp(indep.get_params()[1])),
                           1.5 * truth["sigma"])
        # and fits worse
        self.assertGreater(common.g(common.params), indep.g(indep.params))

    def test_identifiability_needs_constraint(self):
        # an overall shift of the runners' levels is confounded with the
        # intercept
        runners, wt, w, e, X, _ = simulate(tau=0., n_races=300)
        with self.assertRaisesRegex(maxlike.IdentifiabilityError,
                                    "all of 'mu'.*'beta'"):
            fit_without_constraint(runners, wt, w, e, X)


if __name__ == "__main__":
    unittest.main()
