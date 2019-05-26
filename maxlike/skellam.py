import numpy as np
from math import exp
from scipy.special import ndtri, factorial
from scipy.stats.mvn import mvnun


def vectorize(n_in, n_out):
    def wrap(foo):
        return np.frompyfunc(foo, n_in, n_out)
    return wrap


@vectorize(3, 1)
def gauss_bivar(x, y, rho):
    return mvnun(-999 * np.ones((2)), (x, y), (0, 0), np.array([[1, rho], [rho, 1]]))[0]


def poisson_vector(a, size=10):
    curve = np.vectorize(lambda u: a**u / factorial(u))(np.arange(size))
    dcurve = np.append(-1, curve[:-1] - curve[1:])
    return exp(-a) * np.array([curve, dcurve])


def nd_poisson_vector(a, size=10):
    rng = np.arange(size)
    curve = (a[..., None] ** rng) / factorial(rng)
    dcurve = np.insert(curve[..., :-1] - curve[..., 1:], 0, -1, -1)  # arr, pos, value, axis
    return np.exp(-a)[..., None] * np.stack([curve, dcurve])


def nd_poisson_hess(a, size=10):
    rng = np.arange(size)
    curve = (a[..., None] ** rng) / factorial(rng)
    dcurve = np.insert(curve[..., :-1] - curve[..., 1:], 0, -1, -1)
    ddcurve = np.insert(-1, curve[..., :-2] - 2 * curve[..., 1:-1] + curve[..., 2:], 0, -1, -1)
    return exp(-a)[..., None] * np.stack([curve, dcurve, ddcurve])


def skellam_frame(a, b, size=10):
    Poi_a, dPoi_a = poisson_vector(a, size)
    Poi_b, dPoi_b = poisson_vector(b, size)
    sk_frame = [Poi_a[:, None] * Poi_b[None, :]]
    sk_frame += [dPoi_a[:, None] * Poi_b[None, :]]
    sk_frame += [Poi_a[:, None] * dPoi_b[None, :]]
    return sk_frame


def nd_skellam_frame(a, b, size=10):
    # assert a.shape == b.shape
    Poi_a, dPoi_a = nd_poisson_vector(a, size)
    Poi_b, dPoi_b = nd_poisson_vector(b, size)
    sk_frame = [Poi_a[..., None] * Poi_b[..., None, :]]
    sk_frame += [dPoi_a[..., None] * Poi_b[..., None, :]]
    sk_frame += [Poi_a[..., None] * dPoi_b[..., None, :]]
    return sk_frame


def nd_skellam_hess(a, b, size=10):
    Poi_a, dPoi_a, ddPoi_a = nd_poisson_hess(a, size)
    Poi_b, dPoi_b, ddPoi_b = nd_poisson_hess(b, size)
    sk_frame = [Poi_a[..., None] * Poi_b[..., None, :]]
    sk_frame += [dPoi_a[..., None] * Poi_b[..., None, :]]
    sk_frame += [Poi_a[..., None] * dPoi_b[..., None, :]]
    sk_frame += [ddPoi_a[..., None] * Poi_b[..., None, :]]
    sk_frame += [dPoi_a[..., None] * dPoi_b[..., None, :]]
    sk_frame += [Poi_a[..., None] * ddPoi_b[..., None, :]]
    return sk_frame


# scipy.special.pdtr(n, lambda)
def poisson_cdf(a, n):
    # assert isinstance(n, int) and n >= 0
    return poisson_vector(a, n + 1).sum(1)


def poisson_mid_cdf(a, n):
    # assert isinstance(n, int) and n >= 0
    v = poisson_vector(a, n + 1)
    return v[:, :-1].sum(1) + 0.5 * v[:, -1]


def tol2size(tol=1e-8, *args):
    if len(args) == 0:
        args = [1]  # standard value
    return int(-ndtri(tol) * sum(map(lambda x: x ** .5, args)) + sum(args)) + 1


def skellam_cdf(a, b, n=0, tol=1e-8, inverse=False):
    # assert isinstance(n, int)
    size = tol2size(tol, a, b)
    rng = np.arange(size)
    xy = rng[None, :] - rng[:, None]
    xy = (xy >= n) if inverse else (xy <= n)
    return np.asarray(list(map(lambda v: v[xy].sum(), skellam_frame(a, b, size))))


def skellam_cdf_pair(a, b, n=0, tol=1e-8):
    assert isinstance(n, int)
    size = tol2size(tol, a, b)
    rng = np.arange(size)
    xy = rng[None, :] - rng[:, None]
    s = skellam_frame(a, b, size)
    res = np.asarray(list(zip(*[(v[xy < n].sum(), v[xy > n].sum()) for v in s])))
    return res


# pdtri(n, target) = poisson_cdf_root
def poisson_cdf_root(target, n=2, tol=1e-8):
    """
    Finds the parameter 'a' of a Poisson distribution X in order to have
    P(X <= n) = target
    or P(X < n + 1/2 ) = target
    The first guess is estimated through an approximation to the normal
    distribution:
    (n + 1/2 - a) / sqrt(a) = phi^-1(n) = phi
    a + sqrt(a) * phi - n - 1/2 = 0

    Then:
        a = ((sqrt(phi ** 2 + 4 * n + 2) - phi) / 2) ** 2
    """
    # if not (isinstance(target, float) and target < 1 or target > 0):
    if target >= 1 or target <= 0:
        return np.NaN
    phi = ndtri(target)
    min_a = 1e-3
    a = max(((phi * phi + 4 * n + 2) ** .5 - phi) ** 2 / 4, min_a)
    e = 1
    count = 0
    max_steps = 50
    while abs(e) > abs(tol):
        f, df = poisson_cdf(a, n)
        e = target - f
        step = e / df
        scale_factor = 1
        new_a = a + scale_factor * step
        new_f = f
        new_e = abs(target - new_f)
        for i in range(10):
            if new_a > 0 and new_e < abs(e):
                a = new_a
                break
            else:
                new_a = a + scale_factor * step
                new_f, _ = poisson_cdf(new_a, n)
                new_e = abs(target - new_f)
                scale_factor *= .5
        count += 1
        if count > max_steps:
            return np.NaN
    return a


def skellam_cdf_root(target1, target2, n=0, tol=1e-8):
    """
    Finds the parameters (a, b) of a Skellam distribution in order to have
    target1 = P(X-Y < n)
    target2 = P(X-Y > n)
    The first guess is estimated through an approximation to the normal
    distribution
    X-Y ~ N(a-b, a+b)
    then:
        (n - 1/2 - (a-b))/sqrt(a+b) = phi^-1(target1) =: phi_u
        (n + 1/2 - (a-b))/sqrt(a+b) = phi^-1(1-target2) =: phi_d

    n - 1/2 - (a-b) = sqrt(a+b) * phi_u
    n + 1/2 - (a-b) = sqrt(a+b) * phi_d

    sum_ab := 1 / (phi_d - phi_u)^2
    diff_ab := .5 * phi_d / (phi_d - phi_u) - n

    """
    # if not (isinstance(target1, float) and isinstance(target2, float) and
    #        target1 > 0 and target2 > 0 and target1 + target2 < 1):
    if target1 + target2 >= 1 or min(target1, target2) <= 0:
        return [np.NaN, np.NaN]
    phi_d = ndtri(target1)
    phi_u = ndtri(1 - target2)
    sum_ab = 1 / (phi_u - phi_d) ** 2
    diff_ab = 0.5 * (phi_u + phi_d) / (phi_u - phi_d) - n
    floor = 1e-6
    a = max((sum_ab + diff_ab) * 0.5, floor)
    b = max((sum_ab - diff_ab) * 0.5, floor)
    target = np.array([target1, target2])
    e = 1
    count = 0
    max_steps = 50
    while e > abs(tol):
        skellam = skellam_cdf_pair(a, b, n, tol)
        s, ds = skellam[:, 0], skellam[:, [1, 2]]
        e = np.linalg.norm(target - s)
        step = np.linalg.solve(ds, target - s)
        scale_factor = 1
        new_ab = np.maximum((a, b) + scale_factor * step, floor)
        new_s = s
        new_e = np.linalg.norm(new_s - target)
        for i in range(10):
            if new_e < e:
                a, b = new_ab
                break
            else:
                scale_factor *= .5
                new_ab = np.maximum((a, b) + scale_factor * step, floor)
                new_s = skellam_cdf_pair(new_ab[0], new_ab[1], n, tol)[:, 0]
                new_e = np.linalg.norm(new_s - target)
        count += 1
        if count > max_steps:
            return [np.NaN, np.NaN]
    return a, b


def skellam_triangle_root(b, target, tol=1e-8):
    """
    finds the root of
        g(a) = P(X_a < Y_b) + 0.5 * P(X_a == Y_b) - target

    TODO: smart guess
    """
    if target <= 0 or target >= 1:
        return np.NaN
    a = b
    e = 1
    count = 0
    floor = 1e-6
    max_steps = 50
    size = tol2size(.5 * tol, b)
    Poi_b, dPoi_b = poisson_vector(b, size)
    rng_b = np.arange(size)
    while e > abs(tol):
        size = tol2size(tol, a, b)
        rng_a = np.arange(size)
        xy = rng_b[None, :] - rng_a[:, None]
        Poi_a, dPoi_a = poisson_vector(a, size)
        frame = Poi_a[:, None] * Poi_b[None, :]
        dframe = dPoi_a[:, None] * Poi_b[None, :]
        s = frame[xy < 0].sum() + 0.5 * frame[xy == 0].sum()
        ds = dframe[xy < 0].sum() + 0.5 * dframe[xy == 0].sum()
        e = abs(target - s)
        step = (target - s) / ds
        scale_factor = 1
        new_a = np.maximum(a + scale_factor * step, floor)
        new_s = s
        new_e = abs(target - new_s)
        for i in range(10):
            if new_e < e:
                a = new_a
                break
            else:
                scale_factor *= .5
                new_a = np.maximum(a + scale_factor * step, floor)
                Poi_a, _ = poisson_vector(new_a, size)
                frame = Poi_a[:, None] * Poi_b[None, :]
                new_s = frame[xy < 0].sum() + 0.5 * frame[xy == 0].sum()
                new_e = abs(target - new_s)
        if count > max_steps:
            return np.NaN
    return a
