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
from procurement_auction import simulate  # noqa: E402


def small_problem(seed=0):
    rng = np.random.default_rng(seed)
    T, J = 12, 4
    bidders = rng.random((T, J)) < .6
    bidders[np.arange(T), rng.integers(0, J, T)] = True
    winner = np.array([rng.choice(np.flatnonzero(b)) for b in bidders])
    X = np.column_stack([np.ones(T), rng.normal(size=T)])
    log_price = rng.normal(-.2, .2, T)
    return log_price, winner, bidders, X


def fit(firms, log_price, winner, bidders, X, common_factor,
        constraint=True):
    mle = maxlike.LogLikelihood()
    mle.model = FirstOrderStatistic(log_price, winner, bidders, X,
                           common_factor=common_factor)
    if constraint:
        mle.add_constraint(0, Linear([1]))
    mle.add_param(pd.Series(0., index=firms), name="mu")
    mle.add_param(np.log(.2), name="log_sigma")
    mle.add_param(np.array([log_price.mean(), 0.]), name="beta")
    if common_factor:
        mle.add_param(.05, name="tau")
    mle.fit(N=np.ones(len(log_price)))
    return mle


class Test(unittest.TestCase):

    def test_derivatives(self):
        log_price, winner, bidders, X = small_problem()
        rng = np.random.default_rng(1)
        mu = rng.normal(0, .1, bidders.shape[1])
        for common_factor in (False, True):
            f = FirstOrderStatistic(log_price, winner, bidders, X,
                           common_factor=common_factor, nodes=9)
            params = [mu, np.log(.15), np.array([-.1, .05])]
            if common_factor:
                params.append(.12)
            for use_eval in (True, False):
                with self.subTest(common_factor=common_factor,
                                  use_eval=use_eval):
                    self.assertEqual(derivative_errors(
                        f, params, use_eval=use_eval), [])

    def test_independent_bidders_recovered(self):
        firms, lp, w, b, X, truth = simulate(tau=0., n_tenders=1500)
        mle = fit(firms, lp, w, b, X, common_factor=False)
        mu, log_sigma, beta = mle.get_params()
        mu_se, log_sigma_se, beta_se = mle.std_error()
        self.assertLess(np.max(np.abs(mu - truth["mu"]) / mu_se), 4)
        self.assertLess(abs(float(log_sigma) - np.log(truth["sigma"])),
                        4 * float(log_sigma_se))
        np.testing.assert_array_less(np.abs(beta - truth["beta"]),
                                     4 * beta_se)

    def test_common_factor_recovered_and_needed(self):
        firms, lp, w, b, X, truth = simulate(tau=.12, n_tenders=2000)
        indep = fit(firms, lp, w, b, X, common_factor=False)
        common = fit(firms, lp, w, b, X, common_factor=True)

        mu, log_sigma, beta, tau = common.get_params()
        se = common.std_error()
        self.assertLess(abs(abs(float(tau)) - truth["tau"]),
                        4 * float(se[3]))
        self.assertLess(abs(float(log_sigma) - np.log(truth["sigma"])),
                        4 * float(se[1]))
        rmse = lambda m: np.sqrt(np.mean(  # noqa: E731
            (np.ma.getdata(m.get_params()[0]) - truth["mu"]) ** 2))
        # ignoring the common factor biases the firm markups and the spread
        self.assertLess(rmse(common), rmse(indep) / 2)
        self.assertGreater(float(np.exp(indep.get_params()[1])),
                           1.5 * truth["sigma"])
        # and fits worse
        self.assertGreater(common.g(common.params), indep.g(indep.params))

    def test_identifiability_needs_constraint(self):
        # an overall shift of the markups is confounded with the intercept
        firms, lp, w, b, X, _ = simulate(tau=0., n_tenders=300)
        with self.assertRaisesRegex(maxlike.IdentifiabilityError,
                                    "all of 'mu'.*'beta'"):
            fit(firms, lp, w, b, X, common_factor=False, constraint=False)


if __name__ == "__main__":
    unittest.main()
