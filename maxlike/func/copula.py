import numpy as np
from scipy.special import ndtri, ndtr, owens_t
from .func_base import FuncMeta
from ..tensor import Tensor


def no_divwarn(foo):
    def wrap(*args, **kwargs):
        with np.errstate(divide='ignore'):
            return foo(*args, **kwargs)
    return wrap


def gauss_bivar(x, y, rho):
    """
    Standard bivariate normal CDF P(X <= x, Y <= y) with correlation rho.

    Uses the closed form based on Owen's T function (Owen, 1956):
        F(h, k) = (Phi(h) + Phi(k)) / 2 - T(h, a_h) - T(k, a_k) - beta
    with a_h = (k - rho h) / (h sqrt(1 - rho^2)), a_k symmetric, and
    beta = 1/2 when h * k < 0 (or h * k == 0 and h + k < 0), else 0.
    """
    h, k = np.broadcast_arrays(np.asarray(x, dtype=float),
                               np.asarray(y, dtype=float))
    if rho == 0:
        # independence: exact product avoids round-off in the closed form
        return ndtr(h) * ndtr(k)
    if rho >= 1:
        return ndtr(np.minimum(h, k))
    if rho <= -1:
        return np.maximum(ndtr(h) - ndtr(-k), 0)
    # the closed form is singular at h == 0 or k == 0 but continuous,
    # so zeros are nudged to a tiny value
    eps = 1e-15
    h = np.where(h == 0, eps, h)
    k = np.where(k == 0, eps, k)
    s = np.sqrt(1 - rho * rho)
    a_h = (k - rho * h) / (h * s)
    a_k = (h - rho * k) / (k * s)
    beta = np.where(h * k < 0, .5, 0.)
    res = .5 * (ndtr(h) + ndtr(k)) - owens_t(h, a_h) - owens_t(k, a_k) - beta
    return np.clip(res, 0, 1)


def copulaWrap(C):
    """
    Decorator to transform a copula function
    in a function that accepts unnormalized marginal pdf's
    and return a (normalized) joint distribution.
    """
    def wrap(obj, params, *args, **kwargs):
        x = np.asarray(params[0]).cumsum(-1)
        y = np.asarray(params[1]).cumsum(-1)
        x /= x[..., -1][..., None]
        y /= y[..., -1][..., None]
        # exclude last element
        F_xy = C(obj, x[..., None], y[..., None, :], *args, **kwargs)
        F_xy = np.insert(F_xy, 0, 0, -1)
        F_xy = np.insert(F_xy, 0, 0, -2)
        # add 999 at end
        res = np.diff(np.diff(F_xy, 1, -1), 1, -2)
        return Tensor(res)
    return wrap


class CopulaMeta(FuncMeta):
    def __new__(cls, name, bases, attrs, **kwargs):
        res = type.__new__(cls, name, bases, attrs, **kwargs)
        res.__call__ = copulaWrap(res.__call__)
        return res


class Copula(metaclass=CopulaMeta):
    pass


class Gaussian(Copula):
    """
    Receives the marginals of X and Y and returns
    a joint distribution of XY using a gaussian copula.

    -1 < rho < 1

    Note:
    Assuming a Bivariate Gaussian distribution
    tau = (2 / pi) * arcsin(rho)

    where:
    tau := Kendall's tau
    rho := Pearson's Rho

    Degenerates into independence copula when rho=0
    """
    def __init__(self, rho=0):
        assert rho <= 1
        assert rho >= -1
        self.rho = rho

    def __call__(self, x, y):
        return gauss_bivar(
            np.clip(ndtri(x), -999, 999),
            np.clip(ndtri(y), -999, 999),
            self.rho)


class Clayton(Copula):
    """
    C(u,v) = (u^-a + v^-a - 1)^(-1/a)

    Kendall's tau:
    t = a / (2 + a)
    or
    a = 2 t / (1 - t)

    a > 0
    """
    def __init__(self, rho=1):
        assert rho > 0
        self.a = rho

    @no_divwarn
    def __call__(self, x, y):
        return np.power(
            np.power(x, -self.a) + np.power(y, -self.a) - 1,
            -1 / self.a)


class Gumbel(Copula):
    """
    C(u,v) = exp(-((-log u)^a + (-log v)^a)^(1/a))

    Kendall's tau:
    t = 1 - 1 / a
    or
    a = 1 / (1 - t)

    a >= 1

    Degenerates into independence copula when a=1
    """
    def __init__(self, rho=1):
        assert rho >= 1
        self.a = rho

    @no_divwarn
    def __call__(self, x, y):
        return np.exp(-np.power(
            np.power(-np.log(x), self.a) +
            np.power(-np.log(y), self.a),
            1 / self.a))


class Frank(Copula):
    """
    C(u, v) = -(1/a) * log(1 + (e^(-a*u) - 1) * (e^(-a*v) - 1) / (e^-a - 1) )

    a != 0
    """
    def __init__(self, rho=1):
        assert rho != 0
        self.a = rho

    def __call__(self, x, y):
        return (-1 / self.a) * np.log(1 +
            (np.exp(-self.a * x) - 1) *
            (np.exp(-self.a * y) - 1) /
            (np.exp(-self.a) - 1))


class AkiMikhailHaq(Copula):
    """
    Aki-Mikhail-Haq Copula
    C(u,v) = uv / (1 - rho * (1-u) * (1-v))

    -1 < rho < 1

    Degenerates into independence copula when a=1
    """
    def __init__(self, rho=0):
        assert rho <= 1
        assert rho >= -1
        self.rho = rho

    def __call__(self, x, y):
        return x * y / (1 - self.rho * (1 - x) * (1 - y))
