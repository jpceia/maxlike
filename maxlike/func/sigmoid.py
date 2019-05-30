import numpy as np
from ..tensor import Tensor
from .func_base import Func, grad_tensor, hess_tensor
from scipy.special import expit


class Logistic(Func):

    def __call__(self, params):
        return Tensor(expit(params[0]))

    def grad(self, params, i):
        f = expit(params[0])
        return grad_tensor(f * (1 - f), params, i, True)

    def hess(self, params, i, j):
        f = expit(params[0])
        return hess_tensor(f * (1 - f) * (1 - 2 * f),
                           params, i, j, True, True)

    def eval(self, params):
        f = expit(params[0])
        return Tensor(f), \
               [grad_tensor(f * (1 - f), params, 0, True)], \
               [[hess_tensor(f * (1 - f) * (1 - 2 * f),
                             params, 0, 0, True, True)]]

