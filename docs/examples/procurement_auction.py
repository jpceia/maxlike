"""
Procurement auctions where only the winner and the winning price are
observed: recover each firm's bid distribution with two models,

  A. bidders independent given the tender covariates;
  B. a latent tender-level cost factor shared by all bids of a tender,

on synthetic data with a known truth.

    python docs/examples/procurement_auction.py
"""
import numpy as np
import pandas as pd
import maxlike
from maxlike.func import FirstOrderStatistic, Linear


def simulate(n_firms=15, n_tenders=3000, tau=0.12, sigma=0.10, seed=0):
    rng = np.random.default_rng(seed)
    firms = ["F%02d" % j for j in range(n_firms)]
    mu = rng.normal(0, .06, n_firms)
    mu -= mu.mean()                                   # identification
    beta = np.array([-0.15, 0.05])                    # intercept, covariate
    participation = rng.uniform(.15, .45, n_firms)
    X = np.column_stack([np.ones(n_tenders), rng.normal(size=n_tenders)])
    bidders = rng.random((n_tenders, n_firms)) < participation
    # at least two bidders per tender
    for t in np.flatnonzero(bidders.sum(1) < 2):
        bidders[t, rng.choice(n_firms, 2, replace=False)] = True
    c = X @ beta + tau * rng.normal(size=n_tenders)   # tender cost factor
    log_bids = c[:, None] + mu[None, :] + sigma * rng.normal(
        size=(n_tenders, n_firms))
    log_bids[~bidders] = np.inf
    winner = log_bids.argmin(1)
    log_price = log_bids.min(1)
    truth = {"mu": mu, "sigma": sigma, "beta": beta, "tau": tau}
    return firms, log_price, winner, bidders, X, truth


def fit(firms, log_price, winner, bidders, X, common_factor):
    mle = maxlike.LogLikelihood()
    mle.model = FirstOrderStatistic(log_price, winner, bidders, X,
                           common_factor=common_factor)
    mle.add_constraint(0, Linear([1]))                # sum(mu) = 0
    mle.add_param(pd.Series(0., index=firms), name="mu")
    mle.add_param(np.log(.2), name="log_sigma")
    mle.add_param(np.array([log_price.mean(), 0.]), name="beta",
                  labels=[["intercept", "covariate"]])
    if common_factor:
        mle.add_param(.05, name="tau")
    mle.fit(N=np.ones(len(log_price)))
    return mle


def holdout_score(mle, log_price, winner, bidders, X, common_factor):
    """Mean log-likelihood per tender on tenders not used in the fit."""
    model = FirstOrderStatistic(log_price, winner, bidders, X,
                       common_factor=common_factor)
    return float(np.mean(model(mle.params).values))


def report(name, mle, truth, common_factor):
    s = mle.summary()
    rows = []
    est_mu = s.loc["mu", "estimate"].values
    se_mu = s.loc["mu", "std_error"].values
    rows.append(("mu (15 firms): RMSE vs truth",
                 np.sqrt(np.mean((est_mu - truth["mu"]) ** 2)),
                 "share within 2 SE: %.0f%%" % (
                     100 * np.mean(np.abs(est_mu - truth["mu"]) < 2 * se_mu))))
    log_sigma = s.loc[("log_sigma", ""), :]
    rows.append(("sigma", np.exp(log_sigma["estimate"]),
                 "truth %.3f" % truth["sigma"]))
    for k, lab in enumerate(["intercept", "covariate"]):
        r = s.loc[("beta", lab), :]
        rows.append(("beta " + lab, r["estimate"], "truth %.3f, SE %.3f" % (
            truth["beta"][k], r["std_error"])))
    if common_factor:
        r = s.loc[("tau", ""), :]
        rows.append(("tau", abs(r["estimate"]), "truth %.3f, SE %.3f" % (
            truth["tau"], r["std_error"])))
    print("\n" + name)
    for label, value, note in rows:
        print("  %-30s %8.4f   %s" % (label, value, note))


if __name__ == "__main__":
    for tau in (0.12, 0.0):
        firms, lp, w, b, X, truth = simulate(tau=tau, n_tenders=4000)
        train, test = slice(0, 3000), slice(3000, None)
        print("=" * 72)
        print("Synthetic data: true tau = %.2f (common tender factor)" % tau)
        scores = {}
        for cf, name in ((False, "A. independent bidders"),
                         (True, "B. latent common factor")):
            mle = fit(firms, lp[train], w[train], b[train], X[train], cf)
            report(name, mle, truth, cf)
            scores[name] = holdout_score(mle, lp[test], w[test], b[test],
                                         X[test], cf)
        print("\n  hold-out log-likelihood per tender (higher is better):")
        for name, sc in scores.items():
            print("    %-28s %.4f" % (name, sc))
