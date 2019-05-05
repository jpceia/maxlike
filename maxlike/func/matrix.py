from .func_base import Func, grad_tensor, hess_tensor, null_func
from ..tensor import Tensor
from array import array
import numpy as np


class CollapseMatrix(Func):

    hess = null_func

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
        p1_mapping = array('b', range(p1 - 2))
        p1_mapping.append(-1)
        p1_mapping.append(-1)
        idx = tuple([None] * (p1 - 2) + [Ellipsis])
        return Tensor(val[idx], p1=p1, dim=1, p1_mapping=p1_mapping)


class MarkovMatrix(Func):

    def __init__(self, size=8, steps=18, skew=None):
        self.size = size
        self.steps = steps
        if skew is None:
            skew = np.ones((size, size))
        self.skew = skew

    def __call__(self, params):
        return self.eval(params)[0]

    def grad(self, params, i):
        return self.eval(params)[1][i]

    def hess(self, params, i, j):
        return self.eval(params)[2][i][j]

    def eval(self, params):
        s1 = np.asarray(params[0] / self.steps)
        s2 = np.asarray(params[1] / self.steps)
        assert s1.shape == s2.shape
        assert np.all(s1 > 0)
        assert np.all(s2 > 0)
        assert np.all(s1 + s2 < 1)
        M = np.zeros(s1.shape + (self.size, self.size))
        D1 = np.zeros_like(M)
        D2 = np.zeros_like(M)
        H11 = np.zeros_like(M)
        H12 = np.zeros_like(M)
        H22 = np.zeros_like(M)

        M[..., 0, 0] = 1.0
        for step in range(self.steps):
            for idx in range(self.size):
                x = self.size - idx - 1
                x_ = min(x + 1, self.size - 1)
                for jdx in range(self.size):
                    y = self.size - jdx - 1
                    y_ = min(y + 1, self.size - 1)
                    f1 = self.skew[x, y]
                    f2 = self.skew[y, x]
                    m = M[..., x, y].copy()

                    dm1 = D1[..., x, y].copy()
                    dm2 = D2[..., x, y].copy()

                    hm11 = H11[..., x, y].copy()
                    hm12 = H12[..., x, y].copy()
                    hm22 = H22[..., x, y].copy()

                    H11[..., x_, y] += s1 * f1 * hm11 + 2 * f1 * dm1
                    H11[..., x, y_] += s2 * f2 * hm11
                    H11[..., x, y]  -= 2 * f1 * dm1 + (s1 * f1 + s2 * f2) * hm11

                    H12[..., x_, y] += s1 * f1 * hm12 + f1 * dm2
                    H12[..., x, y_] += s2 * f2 * hm12 + f2 * dm1
                    H12[..., x, y]  -= f1 * dm1 + f2 * dm2 + (s1 * f1 + s2 * f2) * hm12

                    H22[..., x_, y] += s1 * f1 * hm22
                    H22[..., x, y_] += s2 * f2 * hm22 + 2 * f2 * dm2
                    H22[..., x, y]  -= 2 * f2 * dm2 + (s1 * f1 + s2 * f2) * hm22

                    D1[..., x_, y] += s1 * f1 * dm1 + f1 * m
                    D1[..., x, y_] += s2 * f2 * dm1
                    D1[..., x, y]  -= f1 * m + (s1 * f1 + s2 * f2) * dm1

                    D2[..., x_, y] += s1 * f1 * dm2
                    D2[..., x, y_] += s2 * f2 * dm2 + f2 * m
                    D2[..., x, y]  -= f2 * m + (s1 * f1 + s2 * f2) * dm2

                    M[..., x_, y] += s1 * f1 * m
                    M[..., x, y_] += s2 * f2 * m
                    M[..., x, y]  -= (s1 * f1 + s2 * f2) * m

        D1 /= self.steps
        D2 /= self.steps
        H11 /= self.steps * self.steps
        H12 /= self.steps * self.steps
        H22 /= self.steps * self.steps

        return Tensor(M, dim=2), \
               [grad_tensor(D1, params, 0, True, dim=2),
                grad_tensor(D2, params, 1, True, dim=2)], \
               [[hess_tensor(H11, params, 0, 0, True, True, dim=2)],
                [hess_tensor(H12, params, 0, 1, True, True, dim=2),
                 hess_tensor(H22, params, 1, 1, True, True, dim=2)]]
