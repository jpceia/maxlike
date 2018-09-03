import numpy as np
from math import exp
from scipy.special import ndtri, factorial
from scipy.stats.mvn import mvnun


def poisson_vector(a, size=10):
    curve = np.vectorize(lambda u: a**u / factorial(u))(np.arange(size))
    dcurve = np.append(-1, curve[:-1] - curve[1:])
    return exp(-a) * np.array([curve, dcurve])


def skellam_frame(a, b, size=10):
    x, dx = poisson_vector(a, size)
    y, dy = poisson_vector(b, size)
    frame = [
        x[:, None] * y[None, :],
        dx[:, None] * y[None, :],
        x[:, None] * dy[None, :],
    ]
    return frame


def tol2size(tol=1e-8, *args):
    if len(args) == 0:
        args = [1]  # standard value
    return int(-ndtri(tol) * sum(map(lambda x: x ** 0.5, args)) + sum(args)) + 1


def skellam_cdf(a, b, n, tol=1e-8, inverse=False):
    # assert isinstance(n, int)
    size = tol2size(tol, a, b)
    rng = np.arange(size)
    xy = rng[None, :] - rng[:, None]
    xy = (xy >= n) if inverse else (xy <= n)
    return np.asarray(map(lambda v: v[xy].sum(), skellam_frame(a, b, size)))


def skellam_cdf_pair(a, b, tol=1e-8):
    size = tol2size(tol, a, b)
    rng = np.arange(size)
    xy = rng[None, :] - rng[:, None]
    s = skellam_frame(a, b, size)
    res = np.asarray(list(zip(*[(v[xy < 0].sum(), v[xy > 0].sum()) for v in s])))
    return res


def skellam_cdf_root(target1, target2, tol=1e-8):
    """
    Finds the parameters (a, b) of a Skellam distribution in order to have
    target1 = P(X-Y < 0)
    target2 = P(X-Y > 0)
    The first guess is estimated through an approximation to the normal
    distribution
    X-Y ~ N(a-b, a+b)
    then:
        (-0.5 - (a-b))/sqrt(a+b) = phi^-1(target1) =: phi_u
        (+0.5 - (a-b))/sqrt(a+b) = phi^-1(1-target2) =: phi_d

    -0.5 - (a-b) = sqrt(a+b) * phi_u
    +0.5 - (a-b) = sqrt(a+b) * phi_d

    sum_ab := 1 / (phi_d - phi_u)^2
    diff_ab := 0.5 * phi_d / (phi_d - phi_u)

    """
    # if not (isinstance(target1, float) and isinstance(target2, float) and
    #        target1 > 0 and target2 > 0 and target1 + target2 < 1):
    if target1 + target2 >= 1 or min(target1, target2) <= 0:
        return [np.NaN, np.NaN]
    phi_d = ndtri(target1)
    phi_u = ndtri(1 - target2)
    sum_ab = 1 / (phi_u - phi_d) ** 2
    diff_ab = .5 * (phi_u + phi_d) / (phi_u - phi_d)
    min_ab = 1e-2
    a = max((sum_ab + diff_ab) / 2, min_ab)
    b = max((sum_ab - diff_ab) / 2, min_ab)
    target = np.array([target1, target2])
    e = 1
    count = 0
    max_steps = 50
    while e > abs(tol):
        skellam = skellam_cdf_pair(a, b, tol)
        s, ds = skellam[:, 0], skellam[:, [1, 2]]
        e = np.linalg.norm(target - s)
        step = np.linalg.solve(ds, target - s)
        scale_factor = 1
        new_ab = (a, b) + scale_factor * step
        new_s = s
        new_e = np.linalg.norm(new_s - target)
        for i in range(10):
            if (all(new_ab > 0) and new_e < e):
                a, b = new_ab
                break
            else:
                new_ab = (a, b) + scale_factor * step
                new_s = skellam_cdf_pair(new_ab[0], new_ab[1], tol)[:, 0]
                new_e = np.linalg.norm(new_s - target)
                scale_factor *= .5
        count += 1
        if count > max_steps:
            return [np.NaN, np.NaN]
    return a, b
