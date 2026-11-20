import numpy as np
from scipy.special import factorial, gammaln
from ..tensor import Tensor
from .func import Exp
from .func_base import Func, grad_tensor, hess_tensor


class Poisson(Func):
    """
    Non normalized
    vector (lambda ^ x) / x!
    """

    def __init__(self, size=10):
        self.size = size

    def _vec(self, params):
        a = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        return (a ** x) / factorial(x)

    @staticmethod
    def _shift(vec):
        # d/da of (a ^ x) / x! is the same vector shifted by one position
        return np.insert(vec[..., :-1], 0, 0, -1)

    def __call__(self, params):
        return Tensor(self._vec(params), dim=1)

    def grad(self, params, i):
        vec = self._shift(self._vec(params))
        return grad_tensor(vec, params, i, True, dim=1)

    def hess(self, params, i, j):
        vec = self._shift(self._shift(self._vec(params)))
        return hess_tensor(vec, params, i, j, True, True, dim=1)

    def eval(self, params):
        vec = self._vec(params)
        val = Tensor(vec, dim=1)
        vec = self._shift(vec)
        grad = grad_tensor(vec, params, 0, True, dim=1)
        vec = self._shift(vec)
        hess = hess_tensor(vec, params, 0, 0, True, True, dim=1)
        return val, [grad], [[hess]]


class LogNegativeBinomial(Func):

    def __init__(self, size=10, r=1):
        self.size = size
        self.r = r

    def _val(self, params):
        """
        gammaln(r + x) - gammaln(r) - ln x! +
        x ln(m) + r ln(r) - (x + r) * ln(r + m))
        """
        m = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        vec = x * np.log(m)
        vec += gammaln(self.r + x)
        vec -= gammaln(self.r)
        vec += self.r * np.log(self.r)
        vec -= np.log(factorial(x))
        vec -= (x + self.r) * np.log(self.r + m)
        return vec

    def _grad(self, params):
        """
        grad_f = (x / m) - (x + r) / (r + m)
        """
        m = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        return x / m - (x + self.r) / (self.r + m)

    def _hess(self, params):
        """
        hess_f = (x + r) / (r + m) ** 2 - x / m ** 2
        """
        m = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        return (x + self.r) / np.square(self.r + m) - x / np.square(m)

    def __call__(self, params):
        return Tensor(self._val(params), dim=1)

    def grad(self, params, i):
        return grad_tensor(self._grad(params), params, i, True, dim=1)

    def hess(self, params, i, j):
        return hess_tensor(self._hess(params), params, i, j, True, True, dim=1)

    def eval(self, params):
        return Tensor(self._val(params), dim=1), \
               [grad_tensor(self._grad(params), params, 0, True, dim=1)], \
               [[hess_tensor(self._hess(params), params, 0, 0,
                             True, True, dim=1)]]


def NegativeBinomial(*args, **kwargs):
    return Exp() @ LogNegativeBinomial(*args, **kwargs)
