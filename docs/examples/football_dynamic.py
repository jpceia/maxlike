"""
Football with dynamic team strengths: attack and defence change over the
season instead of being constant.

    log E[goals of team i against team j in period t] =
        attack[i, t] - defence[j, t] + home * (venue - 1/2)

    attack[i, t] = attack[i, t - 1] + sigma * eps     (Gaussian random walk)
    defence[i, t] = defence[i, t - 1] + sigma * eps

The random walk is a penalty added to the log-likelihood (RandomWalk, a
regularization), so the fit is the posterior mode of the strengths given
sigma: a smoother over the whole season, with exact Newton steps. sigma
sets how fast strengths may move: sigma -> 0 gives back the static model.
It is chosen by forecasting: refit with the data up to each period and
score the goals of the next one, as a model is used in practice.

The data are synthetic, with a known truth (tests/data has no match dates).
Real data only need a `period` column (round, week or month of each
match).

    python docs/examples/football_dynamic.py
"""
import numpy as np
import pandas as pd
from scipy.special import gammaln
import maxlike
from maxlike.func import Take, Vector, Sum, Exp, Linear, RandomWalk


def schedule(n_teams, rng):
    """Double round robin (circle method), venues swapped in the 2nd half."""
    teams = list(range(n_teams))
    rounds = []
    for _ in range(n_teams - 1):
        rounds.append([(teams[k], teams[-1 - k]) for k in range(n_teams // 2)])
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    rounds = [[p if rng.random() < .5 else p[::-1] for p in r] for r in rounds]
    return rounds + [[(b, a) for a, b in r] for r in rounds]


def simulate(n_teams=16, n_seasons=2, sigma=0.05, home=0.25, seed=0):
    """
    n_seasons back-to-back double round robins; the strengths follow a
    random walk with steps of size sigma per round. Returns the matches
    (one row each, with the round as period) and the true strengths.
    """
    rng = np.random.default_rng(seed)
    rounds = sum((schedule(n_teams, rng) for _ in range(n_seasons)), [])
    n_rounds = len(rounds)
    steps = sigma * rng.normal(size=(2, n_teams, n_rounds))
    steps[..., 0] = rng.normal(0, .25, (2, n_teams))     # initial levels
    attack, defence = np.cumsum(steps, axis=-1)
    shift = (attack.mean(0) + defence.mean(0)) / 2       # identification
    attack, defence = attack - shift, defence - shift
    rows = []
    for t, r in enumerate(rounds):
        for i, j in r:
            rows.append((t, i, j,
                         rng.poisson(np.exp(attack[i, t] - defence[j, t] +
                                            home / 2)),
                         rng.poisson(np.exp(attack[j, t] - defence[i, t] -
                                            home / 2))))
    matches = pd.DataFrame(rows, columns=["period", "home", "away",
                                          "home_goals", "away_goals"])
    teams = ["Team %02d" % k for k in range(n_teams)]
    matches["home"] = matches["home"].map(teams.__getitem__)
    matches["away"] = matches["away"].map(teams.__getitem__)
    truth = {"attack": pd.DataFrame(attack, index=teams),
             "defence": pd.DataFrame(defence, index=teams),
             "home": home, "sigma": sigma}
    return matches, truth


def observations(matches, teams):
    """One row per team and match: attacking and defending team, venue."""
    code = {t: k for k, t in enumerate(teams)}
    home = matches["home"].map(code).values
    away = matches["away"].map(code).values
    period = np.tile(matches["period"].values, 2)
    return (np.concatenate([home, away]), np.concatenate([away, home]),
            period, np.repeat([1., 0.], len(matches)),
            np.concatenate([matches["home_goals"],
                            matches["away_goals"]]).astype(float))


def fit(matches, sigma=None, n_periods=None):
    """
    sigma=None: static model, one attack and one defence per team.
    Otherwise attack and defence per team and period (0 .. n_periods - 1;
    periods after the last match are a forecast), tied by a random walk
    with steps of size sigma.
    """
    teams = sorted(set(matches["home"]) | set(matches["away"]))
    attacking, defending, period, venue, goals = observations(matches, teams)
    if n_periods is None:
        n_periods = int(matches["period"].max()) + 1
    dynamic = sigma is not None

    model = Sum(1)
    model.add(Take((attacking, period) if dynamic else attacking), 0, 0)
    model.add(-Take((defending, period) if dynamic else defending), 1, 0)
    model.add(Vector(venue - .5), 2, 0)

    mle = maxlike.Poisson()
    mle.model = model
    # one constraint is enough: the random walk ties the periods together
    mle.add_constraint([0, 1], Linear([1, 1]))
    if dynamic:
        mle.add_regularization([0, 1], RandomWalk(sigma))
        start = pd.DataFrame(0., index=teams, columns=range(n_periods))
    else:
        start = pd.Series(0., index=teams)
    mle.add_param(start, name="attack")
    mle.add_param(start.copy(), name="defence")
    mle.add_param(0., name="home")
    mle.fit(N=np.ones(goals.size), X=goals)
    return mle, teams


def log_score(mle, teams, matches, period):
    """Mean Poisson log-likelihood of the goals of `matches` in `period`."""
    attacking, defending, _, venue, goals = observations(matches, teams)
    a, d, h = [np.ma.getdata(p) for p in mle.get_params()]
    if a.ndim == 2:
        a, d = a[:, period], d[:, period]
    rate = np.exp(a[attacking] - d[defending] + h * (venue - .5))
    return np.mean(goals * np.log(rate) - rate - gammaln(goals + 1))


def forecast_score(matches, sigma, first, step=1):
    """
    One-period-ahead forecasts: for each period t >= first, fit on the
    periods before t (the strengths of period t are the forecast, the
    random walk carries the last ones forward) and score period t.
    """
    scores = []
    for t in range(first, int(matches["period"].max()) + 1, step):
        past = matches[matches["period"] < t]
        mle, teams = fit(past, sigma, n_periods=t + 1)
        scores.append(log_score(mle, teams,
                                matches[matches["period"] == t], t))
    return np.mean(scores)


def choose_sigma(matches, grid, first, step):
    table = pd.Series({s: forecast_score(matches, s, first, step)
                       for s in grid}, name="log score")
    table.index.name = "sigma"
    return table.idxmax(), table


def expected_goals(mle, teams, home_team, away_team, period, horizon=0):
    """
    Expected goals of both sides of a match in `period`, with standard
    errors. horizon > 0 forecasts that many periods after `period`: the
    estimate is the same, its variance grows by 2 * horizon * sigma^2 on
    the log scale (two random walks, attack and defence).
    """
    k, j = teams.index(home_team), teams.index(away_team)
    match = Sum(1)
    match.add(Take(([k, j], [period, period])), 0, 0)
    match.add(-Take(([j, k], [period, period])), 1, 0)
    match.add(Vector([.5, -.5]), 2, 0)
    value, se = mle.predict(Exp() @ match, se=True)
    sigma = mle.reg[0][1].s
    log_se = np.sqrt((se / value) ** 2 + 2 * horizon * sigma ** 2)
    return value, value * log_se


def main():
    matches, truth = simulate()
    n_periods = int(matches["period"].max()) + 1
    print("%d matches, %d rounds; true sigma %.3f per round" % (
        len(matches), n_periods, truth["sigma"]))

    # sigma by one-round-ahead forecasts over the second season
    grid = [0.005, 0.02, 0.05, 0.1, 0.2]
    sigma, table = choose_sigma(matches, grid, first=n_periods // 2, step=3)
    print("\nForecast log score per goal count (higher is better):")
    print(table.round(4).to_string())
    print("chosen sigma: %.3f" % sigma)

    static, teams = fit(matches)
    dynamic, _ = fit(matches, sigma)

    # error against the true strengths of every team and round
    true_a = truth["attack"].values
    a_static = np.ma.getdata(static.get_params()[0])[:, None]
    a_dynamic = np.ma.getdata(dynamic.get_params()[0])
    print("\nRMSE of attack against the truth (all teams and rounds):")
    print("  static  %.3f" % np.sqrt(np.mean((a_static - true_a) ** 2)))
    print("  dynamic %.3f" % np.sqrt(np.mean((a_dynamic - true_a) ** 2)))

    s = dynamic.summary()
    home = s.loc[("home", "")]
    print("\nHome advantage: %.3f +/- %.3f (true %.2f)" % (
        home["estimate"], home["std_error"], truth["home"]))

    # who moved the most over the two seasons
    attack = dynamic.get_params(labeled=True)[0]
    change = (attack.iloc[:, -1] - attack.iloc[:, 0]).sort_values()
    print("\nLargest changes in attack, first to last round (true change):")
    true_change = true_a[:, -1] - true_a[:, 0]
    for team in list(change.index[:2]) + list(change.index[-2:]):
        print("  %s %+.2f (%+.2f)" % (team, change[team],
                                      true_change[teams.index(team)]))

    last = n_periods - 1
    strength = attack.iloc[:, -1] + \
        dynamic.get_params(labeled=True)[1].iloc[:, -1]
    best, worst = strength.idxmax(), strength.idxmin()
    value, se = expected_goals(dynamic, teams, best, worst, last, horizon=1)
    print("\nNext round, %s (home) v %s:" % (best, worst))
    print("  %s %.2f +/- %.2f   %s %.2f +/- %.2f" % (
        best, value[0], se[0], worst, value[1], se[1]))
    return dynamic, static, truth


if __name__ == "__main__":
    main()
