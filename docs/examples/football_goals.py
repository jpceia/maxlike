"""
Football: attack and defence strength of each team from match scores.

    log E[goals of team i against team j] = attack[i] - defence[j]
                                            + home * (venue - 1/2)

Data come one row per match (tests/data/data2.csv: home team, away team,
home goals, away goals). Each match gives two observations, one per side,
gathered from the parameter vectors with Take: no team x team grid.

    python docs/examples/football_goals.py
"""
import os
import numpy as np
import pandas as pd
import maxlike
from maxlike.func import Take, Vector, Sum, Exp, Linear

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "tests", "data", "data2.csv")


def load(path=DATA):
    matches = pd.read_csv(path)
    matches["home"] = "Team " + matches["t1"].map("{:02d}".format)
    matches["away"] = "Team " + matches["t2"].map("{:02d}".format)
    return matches[["home", "away", "g1", "g2"]].rename(
        columns={"g1": "home_goals", "g2": "away_goals"})


def fit(matches):
    teams = sorted(set(matches["home"]) | set(matches["away"]))
    code = {t: k for k, t in enumerate(teams)}
    home, away = matches["home"].map(code).values, matches["away"].map(code).values
    # one observation per team and match: (attacking team, defending team)
    attacking = np.concatenate([home, away])
    defending = np.concatenate([away, home])
    venue = np.repeat([1., 0.], len(matches))       # 1 = playing at home
    goals = np.concatenate([matches["home_goals"], matches["away_goals"]])

    model = Sum(1)
    model.add(Take(attacking), 0, 0)
    model.add(-Take(defending), 1, 0)
    model.add(Vector(venue - .5), 2, 0)

    mle = maxlike.Poisson()
    mle.model = model
    mle.add_constraint([0, 1], Linear([1, 1]))       # identification
    mle.add_param(pd.Series(0., index=teams), name="attack")
    mle.add_param(pd.Series(0., index=teams), name="defence")
    mle.add_param(0., name="home")
    mle.fit(N=np.ones(goals.size), X=goals.astype(float))
    return mle, teams


def expected_goals(mle, teams, home_team, away_team):
    """Expected goals of both sides of one match, with standard errors."""
    k, j = teams.index(home_team), teams.index(away_team)
    match = Sum(1)
    match.add(Take([k, j]), 0, 0)
    match.add(-Take([j, k]), 1, 0)
    match.add(Vector([.5, -.5]), 2, 0)
    return mle.predict(Exp() @ match, se=True)


def main():
    matches = load()
    mle, teams = fit(matches)

    s = mle.summary()
    table = pd.DataFrame({
        "attack": s.loc["attack", "estimate"],
        "attack_se": s.loc["attack", "std_error"],
        "defence": s.loc["defence", "estimate"],
        "defence_se": s.loc["defence", "std_error"],
    })
    table["strength"] = table["attack"] + table["defence"]
    print("%d matches, %d teams" % (len(matches), len(teams)))
    print("\nStrongest teams (attack + defence, log scale):")
    print(table.sort_values("strength", ascending=False).head(5).round(3))
    home = s.loc[("home", "")]
    print("\nHome advantage: %.3f +/- %.3f (x%.2f goals)" % (
        home["estimate"], home["std_error"], np.exp(home["estimate"])))

    best, worst = table["strength"].idxmax(), table["strength"].idxmin()
    value, se = expected_goals(mle, teams, best, worst)
    print("\nExpected goals, %s (home) v %s:" % (best, worst))
    print("  %s %.2f +/- %.2f   %s %.2f +/- %.2f" % (
        best, value[0], se[0], worst, value[1], se[1]))
    return mle


if __name__ == "__main__":
    main()
