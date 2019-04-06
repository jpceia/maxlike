import numpy as np
from ..tensor import Tensor
from .func_base import Func, grad_tensor, hess_tensor, null_func


class Linear(Func):

    def __init__(self, weight_list=None):
        assert isinstance(weight_list, (list, tuple))
        self.weights = weight_list

    def __call__(self, params):
        return sum(((w * np.asarray(p)).sum()
                    for w, p in zip(params, self.weights)))

    def grad(self, params, i):
        return grad_tensor(self.weights[i] * np.ones((1, ) *
                           np.asarray(params[i]).ndim), params, i)

    def hess(self, params, i, j):
        return hess_tensor(np.zeros((1, ) *
            (np.asarray(params[j]).ndim + np.asarray(params[i]).ndim)),
            params, i, j)


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
            2 * np.ones(params[i].shape) / self.s / self.s,
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

    @staticmethod
    def __val(params):
        return np.exp(np.asarray(params[0]))

    def __call__(self, params):
        return Tensor(Exp.__val(params))

    def grad(self, params, i):
        return grad_tensor(Exp.__val(params), params, i, True)

    def hess(self, params, i, j):
        return hess_tensor(Exp.__val(params), params, i, j, True, True)

    def eval(self, params):
        f = Exp.__val(params)
        return Tensor(f), \
               [grad_tensor(f, params, 0, True)], \
               [[hess_tensor(f, params, 0, 0, True, True)]]


class Logistic(Func):

    @staticmethod
    def __val(params):
        return 1 / (1 + np.exp(-params[0]))

    def __call__(self, params):
        return Tensor(Logistic.__val(params))

    def grad(self, params, i):
        f = Logistic.__val(params)
        return grad_tensor(f * (1 - f), params, i, True)

    def hess(self, params, i, j):
        f = Logistic.__val(params)
        return hess_tensor(f * (1 - f) * (1 - 2 * f),
                           params, i, j, True, True)

    def eval(self, params):
        f = Logistic.__val(params)
        return Tensor(f),\
               [grad_tensor(f * (1 - f), params, 0, True)], \
               [[hess_tensor(f * (1 - f) * (1 - 2 * f),
                             params, 0, 0, True, True)]]

