import numpy as np
from scipy.special import factorial, gammaln
from array import array
from ..tensor import Tensor
from .func_base import Func, grad_tensor, hess_tensor


class Poisson(Func):
    """
    Non normalized
    vector (lambda ^ x) / x!
    """

    def __init__(self, size=10):
        self.size = size

    def __call__(self, params):
        a = np.asarray(params[0])
        rng = np.arange(self.size)
        vec = (a[..., None] ** rng) / factorial(rng)
        return Tensor(vec, dim=1)

    def grad(self, params, i):
        a = np.asarray(params[0])
        rng = np.arange(self.size)
        vec = ((a[..., None] ** rng) / factorial(rng))
        vec = np.insert(vec[..., :-1], 0, 0, -1)
        return grad_tensor(vec, params, i, True, dim=1)

    def hess(self, params, i, j):
        a = np.asarray(params[0])
        rng = np.arange(self.size)
        vec = ((a[..., None] ** rng) / factorial(rng))
        vec = np.insert(vec[..., :-1], 0, 0, -1)
        vec = np.insert(vec[..., :-1], 0, 0, -1)
        return hess_tensor(vec, params, i, j, True, True, dim=1)

    def eval(self, params):
        a = np.asarray(params[0])
        rng = np.arange(self.size)
        vec = (a[..., None] ** rng) / factorial(rng)
        val = Tensor(vec, dim=1)
        vec = np.insert(vec[..., :-1], 0, 0, -1)
        grad = grad_tensor(vec, params, 0, True, dim=1)
        vec = np.insert(vec[..., :-1], 0, 0, -1)
        hess = hess_tensor(vec, params, 0, 0, True, True, dim=1)
        return val, [grad], [[hess]]


class NegativeBinomial(Func):

    def __init__(self, size=10, r=1):
        self.size = size
        self.r = r

    def __call__(self, params):
        """
        exp(
            gammaln(r + x) - gammaln(r) - ln x! +
            x ln(m) + r ln(r) - (x + r) * ln(r + m))
            )
        """
        m = np.asarray(params[0])
        x = np.arange(self.size)
        vec = gammaln(self.r + x) - gammaln(self.r)
        vec -=np.log(factorial(x))
        vec += x * np.log(m) + self.r * np.log(self.r)
        vec -= (x + self.r) * np.log(self.r + m)
        return Tensor(np.exp(vec), dim=1)

    def grad(self, params, i):
        """
        grad_f = ((x / m) - (x + r) / (r + m)) * f
        """
        m = np.asarray(params[0])
        x = np.arange(self.size)
        vec = gammaln(self.r + x) - gammaln(self.r)
        vec -=np.log(factorial(x))
        vec += x * np.log(m) + self.r * np.log(self.r)
        vec -= (x + self.r) * np.log(self.r + m)
        return grad_tensor(
            np.exp(vec) * ((x / m) - (x + self.r) / (self.r + m)),
            params, i, True, dim=1)
