import numpy as np
from scipy.special import factorial, gammaln
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
        vec = ((a[..., None] ** rng) / factorial(rng))[..., :-1]
        vec = np.insert(vec, 0, 0, -1)
        return grad_tensor(vec, params, i, True, dim=1)

    def hess(self, params, i, j):
        a = np.asarray(params[0])
        rng = np.arange(self.size)
        vec = ((a[..., None] ** rng) / factorial(rng))[..., :-2]
        vec = np.insert(vec, 0, 0, -1)
        vec = np.insert(vec, 0, 0, -1)
        return hess_tensor(vec, params, i, j, True, True, dim=1)


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


class CollapseMatrix(Func):

    def __init__(self, conditions=None):
        """
        Condition or list of conditions with the form
        sgn(A*x + B*y + C) == s
        """
        if conditions is None:
            self.conditions = [
                (1, -1, 0, 1),
                (1, -1, 0, 0),
                (1, -1, 0, -1),
            ]

    def __call__(self, params):
        """
        CollapseMatrix function assumes that there is just one param that is
        a Tensor with dim=2 (frame)
        """
        arr = np.asarray(params[0])
        rng_x = np.arange(arr.shape[-2])
        rng_y = np.arange(arr.shape[-1])
        val = []
        for x, y, c, s in self.conditions:
            filt = np.sign(x * rng_x[:, None] +
                           y * rng_y[None, :] + c) == s
            val.append((arr * filt).sum((-1, -2)))
        val = np.stack(val, -1)
        return Tensor(val, dim=1)

    def grad(self, params, i):
        ones = np.ones(np.asarray(params[0]).shape)
        rng_x = np.arange(ones.shape[-2])
        rng_y = np.arange(ones.shape[-1])
        val = []
        for x, y, c, s in self.conditions:
            filt = np.sign(x * rng_x[:, None] +
                           y * rng_y[None, :] + c) == s
            val.append(ones * filt)
        p1 = ones.ndim
        val = np.stack(val, -1)
        val = val.swapaxes(0, p1 - 2)
        val = val.swapaxes(1, p1 - 1)
        p1_mapping = list(range(p1 - 2)) + [-1, -1]
        idx = tuple([None] * (p1 - 2) + [Ellipsis])
        return Tensor(val[idx], p1=p1, dim=1, p1_mapping=p1_mapping)

    def hess(self, params, i, j):
        return Tensor()
