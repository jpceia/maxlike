"""
Races where only the winner and the winning time are observed: recover
each runner's level with two models,

  A. runners independent given the course (observed covariates);
  B. a latent race-day factor (wind, heat, ...) shared by all runners of
     a race,

on synthetic data with a known truth.

    python docs/examples/race.py
"""
import numpy as np
import pandas as pd
import maxlike
from maxlike.func import FirstOrderStatistic, Linear


def simulate(n_runners=15, n_races=3000, tau=0.06, sigma=0.04, seed=0):
    """
    log time[race, runner] = c[race] + mu[runner] + sigma * eps
    c[race] = beta_0 + beta_1 * elevation[race] + tau * u[race]

    Times are relative to a reference time for the distance (log scale),
    elevation is a standardised measure of how hilly the course is, and u
    the unobserved conditions of the day.
    """
    rng = np.random.default_rng(seed)
    runners = ["R%02d" % j for j in range(n_runners)]
    mu = rng.normal(0, .03, n_runners)
    mu -= mu.mean()                                   # identification
    beta = np.array([0.10, 0.03])                     # intercept, elevation
    participation = rng.uniform(.2, .5, n_runners)
    X = np.column_stack([np.ones(n_races), rng.normal(size=n_races)])
    entrants = rng.random((n_races, n_runners)) < participation
    # at least two runners per race
    for t in np.flatnonzero(entrants.sum(1) < 2):
        entrants[t, rng.choice(n_runners, 2, replace=False)] = True
    c = X @ beta + tau * rng.normal(size=n_races)     # race-day factor
    log_times = c[:, None] + mu[None, :] + sigma * rng.normal(
        size=(n_races, n_runners))
    log_times[~entrants] = np.inf
    winner = log_times.argmin(1)
    winning_time = log_times.min(1)
    truth = {"mu": mu, "sigma": sigma, "beta": beta, "tau": tau}
    return runners, winning_time, winner, entrants, X, truth


def fit(runners, winning_time, winner, entrants, X, common_factor):
    """
    Model A starts from a flat guess. Model B starts from model A's fit:
    A's dispersion approximates sqrt(sigma^2 + tau^2), so sigma and tau
    both start at sigma_A / sqrt(2). This keeps tau away from 0, a saddle
    point of the likelihood (it is symmetric in tau).
    """
    if common_factor:
        start = fit(runners, winning_time, winner, entrants, X, False)
        mu0, log_sigma0, beta0 = [np.ma.getdata(p) for p in
                                  start.get_params()]
        log_sigma0 = float(log_sigma0) - .5 * np.log(2)
    else:
        mu0, log_sigma0 = np.zeros(len(runners)), np.log(.1)
        beta0 = np.array([winning_time.mean(), 0.])

    mle = maxlike.LogLikelihood()
    mle.model = FirstOrderStatistic(winning_time, winner, entrants, X,
                                    common_factor=common_factor)
    mle.add_constraint(0, Linear([1]))                # sum(mu) = 0
    mle.add_param(pd.Series(mu0, index=runners), name="mu")
    mle.add_param(log_sigma0, name="log_sigma")
    mle.add_param(beta0, name="beta", labels=[["intercept", "elevation"]])
    if common_factor:
        mle.add_param(np.exp(log_sigma0), name="tau")
    mle.fit(N=np.ones(len(winning_time)))
    return mle


def holdout_score(mle, winning_time, winner, entrants, X, common_factor):
    """Mean log-likelihood per race on races not used in the fit."""
    model = FirstOrderStatistic(winning_time, winner, entrants, X,
                                common_factor=common_factor)
    return float(np.mean(model(mle.params).values))


def report(name, mle, truth, common_factor):
    s = mle.summary()
    rows = []
    est_mu = s.loc["mu", "estimate"].values
    se_mu = s.loc["mu", "std_error"].values
    rows.append(("mu (15 runners): RMSE vs truth",
                 np.sqrt(np.mean((est_mu - truth["mu"]) ** 2)),
                 "share within 2 SE: %.0f%%" % (
                     100 * np.mean(np.abs(est_mu - truth["mu"]) < 2 * se_mu))))
    log_sigma = s.loc[("log_sigma", ""), :]
    rows.append(("sigma", np.exp(log_sigma["estimate"]),
                 "truth %.3f" % truth["sigma"]))
    for k, lab in enumerate(["intercept", "elevation"]):
        r = s.loc[("beta", lab), :]
        rows.append(("beta " + lab, r["estimate"], "truth %.3f, SE %.3f" % (
            truth["beta"][k], r["std_error"])))
    if common_factor:
        r = s.loc[("tau", ""), :]
        rows.append(("tau", abs(r["estimate"]), "truth %.3f, SE %.3f" % (
            truth["tau"], r["std_error"])))
    print("\n" + name)
    for label, value, note in rows:
        print("  %-32s %8.4f   %s" % (label, value, note))


if __name__ == "__main__":
    for tau in (0.06, 0.0):
        runners, wt, w, e, X, truth = simulate(tau=tau, n_races=4000)
        train, test = slice(0, 3000), slice(3000, None)
        print("=" * 72)
        print("Synthetic data: true tau = %.2f (race-day factor)" % tau)
        scores = {}
        for cf, name in ((False, "A. independent runners"),
                         (True, "B. latent race-day factor")):
            mle = fit(runners, wt[train], w[train], e[train], X[train], cf)
            report(name, mle, truth, cf)
            scores[name] = holdout_score(mle, wt[test], w[test], e[test],
                                         X[test], cf)
        print("\n  hold-out log-likelihood per race (higher is better):")
        for name, sc in scores.items():
            print("    %-28s %.4f" % (name, sc))
