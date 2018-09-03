import sys
sys.path.insert(0, r"D:\git\maxlike")
from maxlike.func_base import Func
from maxlike.tensor import Tensor, grad_tensor
import numpy as np

class FiniteMatrix(Func):
    def __init__(self, size=8, steps=18):
        self.size = size
        self.steps = steps

    def __call__(self, params):
        s1 = np.asarray(params[0] / self.steps)
        s2 = np.asarray(params[1] / self.steps)
        assert np.all(s1 >= 0)
        assert np.all(s2 >= 0)
        assert np.all(s1 + s2 < 1)
        M = np.zeros(s1.shape + (self.size, self.size))
        M[..., 0, 0] = 1
        idx = [..., None, None]
        s1 = s1[idx]
        s2 = s2[idx]
        for t in range(self.steps):
            D1 = np.zeros_like(M)
            D2 = np.zeros_like(M)
            for ii in range(self.size):
                for jj in range(self.size):
                    m = M[..., ii, jj]
                    if ii < self.size - 1:
                        D1[..., ii, jj] -= m
                        D1[..., ii + 1, jj] += m
                    if jj < self.size - 1:
                        D2[..., ii, jj] -= m
                        D2[..., ii, jj + 1] += m
            M += s1 * D1 + s2 * D2
        return Tensor(M, dim=2)

    def grad(self, params, i):
        
        s1 = np.asarray(params[0] / self.steps)
        s2 = np.asarray(params[1] / self.steps)
        assert np.all(s1 >= 0)
        assert np.all(s2 >= 0)
        assert np.all(s1 + s2 < 1)
        M = np.zeros(s1.shape + (self.size, self.size))
        DM = np.zeros_like(M)
        M[..., 0, 0] = 1
        idx = [..., None, None]
        s1 = s1[idx]
        s2 = s2[idx]
        for t in range(self.steps):
            D1 = np.zeros_like(M)
            D2 = np.zeros_like(M)
            DD1 = np.zeros_like(M)
            DD2 = np.zeros_like(M)
            for ii in range(self.size):
                for jj in range(self.size):
                    m = M[..., ii, jj]
                    d = DM[..., ii, jj]
                    if ii < self.size - 1:
                        D1[..., ii, jj] -= m
                        D1[..., ii + 1, jj] += m
                        DD1[..., ii, jj] -= d
                        DD1[..., ii + 1, jj] += d
                    if jj < self.size - 1:
                        D2[..., ii, jj] -= m
                        D2[..., ii, jj + 1] += m
                        DD2[..., ii, jj] -= d
                        DD2[..., ii, jj + 1] += d
            M += s1 * D1 + s2 * D2
            DM += D1 if i == 0 else D2
            DM += s1 * DD1 + s2 * DD2
        return grad_tensor(DM, params, i, dim=2)

    def hess(self, params, i, j):
        raise NotImplementedError
