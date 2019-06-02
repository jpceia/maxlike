import numpy as np
from scipy.special import factorial, gammaln
from array import array
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

    def __call__(self, params):
        a = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        vec = (a ** x) / factorial(x)
        return Tensor(vec, dim=1)

    def grad(self, params, i):
        a = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        vec = ((a ** x) / factorial(x))
        vec = np.insert(vec[..., :-1], 0, 0, -1)
        return grad_tensor(vec, params, i, True, dim=1)

    def hess(self, params, i, j):
        a = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        vec = ((a ** x) / factorial(x))
        vec = np.insert(vec[..., :-1], 0, 0, -1)
        vec = np.insert(vec[..., :-1], 0, 0, -1)
        return hess_tensor(vec, params, i, j, True, True, dim=1)

    def eval(self, params):
        a = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        vec = (a ** x) / factorial(x)
        val = Tensor(vec, dim=1)
        vec = np.insert(vec[..., :-1], 0, 0, -1)
        grad = grad_tensor(vec, params, 0, True, dim=1)
        vec = np.insert(vec[..., :-1], 0, 0, -1)
        hess = hess_tensor(vec, params, 0, 0, True, True, dim=1)
        return val, [grad], [[hess]]


class LogNegativeBinomial(Func):

    def __init__(self, size=10, r=1):
        self.size = size
        self.r = r

    def __call__(self, params):
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
        return Tensor(vec, dim=1)

    def grad(self, params, i):
        """
        grad_f = (x / m) - (x + r) / (r + m)
        """
        m = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        return grad_tensor(
            x / m - (x + self.r) / (self.r + m),
            params, i, True, dim=1)

    def hess(self, params, i, j):
        """
        hess_f = (x + r) / (r + m) ** 2 - x / m ** 2
        """
        m = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        return hess_tensor(
            (x + self.r) / np.square(self.r + m) - x / np.square(m),
            params, i, j, True, True, dim=1)

    def eval(self, params):
        m = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        vec = x * np.log(m)
        vec += gammaln(self.r + x)
        vec -= gammaln(self.r)
        vec += self.r * np.log(self.r)
        vec -= np.log(factorial(x))
        vec -= (x + self.r) * np.log(self.r + m)
        val = Tensor(vec, dim=1)
        grad = grad_tensor(
            x / m - (x + self.r) / (self.r + m),
            params, 0, True, dim=1)
        hess = hess_tensor(
            (x + self.r) / np.square(self.r + m) - x / np.square(m),
            params, 0, 0, True, True, dim=1)
        return val, [grad], [[hess]]


def NegativeBinomial(*args, **kwargs):
    return Exp() @ LogNegativeBinomial(*args, **kwargs)
