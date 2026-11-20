import numpy as np
from ..tensor import Tensor
from .func_base import Func, grad_tensor, hess_tensor, null_func


class Linear(Func):
    """
    Linear(w)(p_0, ..., p_k) = sum_i w_i * sum(p_i)

    A scalar function (no feature axes): it can be used as a constraint,
    a regularization or composed with other functions.
    """

    def __init__(self, weight_list=None):
        if isinstance(weight_list, (int, float)):
            weight_list = [weight_list]
        else:
            assert isinstance(weight_list, (list, tuple))
        self.weights = weight_list

    def __call__(self, params):
        return Tensor(sum((w * np.asarray(p)).sum()
                          for w, p in zip(self.weights, params)))

    def grad(self, params, i):
        # dense: every entry of params[i] contributes to the scalar output
        p = np.asarray(params[i])
        return Tensor(np.full(p.shape, self.weights[i], dtype=float),
                      x_dim=p.ndim)

    hess = null_func


class Quadratic(Func):  # TODO : expand this class to allow more generic stuff

    def __init__(self, u=0, s=1):
        self.u = u
        self.s = s

    def __call__(self, params):
        val = 0
        for p in params:
            z = (np.asarray(p) - self.u) / self.s
            val += (z * z).sum()
        return Tensor(val)

    def grad(self, params, i):
        z = (np.asarray(params[i]) - self.u) / self.s
        return grad_tensor(2 * z / self.s, params, i, True).sum()

    def hess(self, params, i, j):
        if i != j:
            return Tensor(0)
        return hess_tensor(
            2 * np.ones_like(params[i]) / self.s / self.s,
            params, i, j, True, True).sum()


class X(Func):

    def __call__(self, params):
        return Tensor(params[0])

    def grad(self, params, i):
        return grad_tensor(np.ones_like(params[0]), params, i, True)

    hess = null_func


class Constant(Func):

    def __init__(self, vector):
        self.vector = np.asarray(vector)

    def __call__(self, params):
        return Tensor(self.vector)

    grad = null_func
    hess = null_func


class Scalar(Func):

    def __call__(self, params):
        return Tensor(params[0])

    def grad(self, params, i):
        return Tensor(1)

    hess = null_func


class Vector(Func):

    def __init__(self, vector):
        self.vector = np.asarray(vector)

    def __call__(self, params):
        return Tensor(np.asarray(params[0]) * self.vector)

    def grad(self, params, i):
        return grad_tensor(self.vector, params, i, None)

    hess = null_func


class Exp(Func):

    def __call__(self, params):
        return Tensor(np.exp(params[0]))

    def grad(self, params, i):
        f = Tensor(np.exp(params[0]))
        return grad_tensor(f, params, i, True, dim=f.dim)

    def hess(self, params, i, j):
        f = Tensor(np.exp(params[0]))
        return hess_tensor(f, params, i, j, True, True, dim=f.dim)

    def eval(self, params):
        val = Tensor(np.exp(params[0]))
        grad = grad_tensor(val, params, 0, True, dim=val.dim)
        hess = hess_tensor(val, params, 0, 0, True, True, dim=val.dim)
        return val, [grad], [[hess]]


class Log(Func):

    def __call__(self, params):
        return Tensor(np.log(params[0]))

    def grad(self, params, i):
        f = Tensor(1 / params[0])
        return grad_tensor(f, params, i, True, dim=f.dim)

    def hess(self, params, i, j):
        f = Tensor(-1 / np.square(params[0]))
        return hess_tensor(f, params, i, j, True, True, dim=f.dim)
