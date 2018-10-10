import numpy as np
from scipy.special import ndtri
from scipy.stats.mvn import mvnun
from six import with_metaclass
from .func_base import FuncMeta
from ..tensor import Tensor


def no_divwarn(foo):
    def wrap(*args, **kwargs):
        with np.errstate(divide='ignore'):
            return foo(*args, **kwargs)
    return wrap


def vectorize(n_in, n_out):
    def wrap(foo):
        return np.frompyfunc(foo, n_in, n_out)
    return wrap


@no_divwarn
@vectorize(3, 1)
def gauss_bivar(x, y, rho):
    return mvnun((-999, -999), (x, y), (0, 0), ((1, rho), (rho, 1)))[0]


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

class Copula(with_metaclass(CopulaMeta)):
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
    """
    def __init__(self, rho=0):
        assert rho <= 1
        assert rho >= -1
        self.rho = rho

    def __call__(self, x, y):
        return gauss_bivar(ndtri(x), ndtri(y), self.rho)


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
    """
    def __init__(self, rho=1):
        assert rho >= 1
        self.a = rho

    def __call__(self, x, y):
        return np.exp(-np.power(
            np.power(-np.log(x), self.a) + np.power(-np.log(y), self.a),
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
    """
    def __init__(self, rho=0):
        assert rho <= 1
        assert rho >= -1
        self.rho = rho

    def __call__(self, x, y):
        return x * y / (1 - self.rho * (1 - x) * (1 - y))
