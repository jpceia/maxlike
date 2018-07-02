from func import FuncMeta, Func, Tensor
from six import with_metaclass
from common import *


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
        F_xy = C(obj, x[..., None, :], y[..., None], *args, **kwargs)
        F_xy = np.insert(F_xy, 0, 0, -1)
        F_xy = np.insert(F_xy, 0, 0, -2)
        res = np.diff(np.diff(F_xy, 1, -1), 1, -2)
        # normalization ?
        return Tensor(res)
    return wrap


class CopulaMeta(FuncMeta):
    def __new__(cls, name, bases, attrs, **kwargs):
        res = type.__new__(cls, name, bases, attrs, **kwargs)
        res.__call__ = copulaWrap(res.__call__)
        return res

class Copula(Func, with_metaclass(CopulaMeta, object)):
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
    def __init__(self, rho):
        assert rho < 1
        assert rho > -1
        self.rho = rho

    def __call__(self, x, y):
        return gauss_bivar(ndtri(x), ndtri(y), self.rho)


class TStudent(Copula):
    """
    -1 < rho < 1
    v > 0
    """
    def __init__(self, rho, v):
        assert rho < 1
        assert rho > -1
        assert v > 0
        self.rho = rho
        self.v = v

    def __call__(self, x, y):
        u = stdtridf(x, self.v)
        v = stdtridf(y, self.v)
        u[..., -1] = 999
        v[..., -1, :] = 999
        t = np.sqrt((u * u + v * v - 2 * self.rho * u * v) /
                    (1 - self.rho * self.rho))
        return stdtr(t, self.v)


class Clayton(Copula):
    """
    C(u,v) = (u^-a + v^-a - 1)^(-1/a)

    Kendall's tau:
    t = a / (2 + a)
    or
    a = 2 t / (1 - t)

    a > 0
    """
    def __init__(self, a):
        assert a > 0
        self.a = a

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
    def __init__(self, a):
        assert a >= 1
        self.a = a

    def __call__(self, x, y):
        return np.exp(-np.power(
            np.power(-np.log(x), self.a) + np.power(-np.log(y), self.a),
            1 / self.a))


class Frank(Copula):
    """
    C(u, v) = -(1/a) * log(1 + (e^(-a*u) - 1) * (e^(-a*v) - 1) / (e^-a - 1) )

    a != 0
    """
    def __init__(self, a):
        assert a != 0
        self.a = a

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
    def __init__(self, rho):
        assert rho < 1
        assert rho > -1
        self.rho = rho

    def __call__(self, x, y):
        return x * y / (1 - self.rho * (1 - x) * (1 - y))
