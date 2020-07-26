from .func_base import Func, grad_tensor, hess_tensor
from ..tensor import Tensor
import numpy as np


class Game(Func):
    """
    Func with probabilities of winning a (tennis) gamme,
    given the probability of winning a point.
    """
    
    def __call__(self, params):
        p = np.asarray(params[0])
        q = 1 - p
        g = []
        g.append(p ** 4)
        g.append(4 * p ** 4 * q)
        g.append(10 * p ** 4 * q ** 2)
        g.append(20 * p ** 5 * q ** 3 / (1 - 2 * p * q))
        return Tensor(np.stack(g).swapaxes(0, -1), dim=1)
    
    def grad(self, params, i):
        p = np.asarray(params[0])
        q = 1 - p
        d = []
        d.append(4 * p ** 3)
        d.append(4 * (4 - 5 * p) * p ** 3)
        d.append(20 * (2 - 3 * p) * q * p ** 3)
        d.append(-20 * (12 * p ** 3 - 20 * p ** 2 + 16 * p - 5) *
                 q ** 2 * p ** 4 / (1 - 2 * p * q) ** 2)
        return grad_tensor(np.stack(d).swapaxes(0, -1),
                           params, i, True, dim=1)
    
    def hess(self, params, i, j):
        p = np.asarray(params[0])
        q = 1 - p
        h = []
        h.append(12 * p ** 2)
        h.append(16 * (3 - 5 * p) * p ** 2)
        h.append(20 * (5 * (1 - 3 * p) * q + 1) * p ** 2)
        h.append(40 * (60 * p ** 6 - 200 * p ** 5 + 322 * p ** 4 -
                       308 * p ** 3 + 184 * p ** 2 - 65 * p + 10) *
                 q * p ** 3 / (1 - 2 * p * q) ** 3)
        return hess_tensor(np.stack(h).swapaxes(0, -1),
                           params, i, j, True, True, dim=1)


class TieBreak(Func):
    """
    Func with probabilities of winning a (tennis) tiebreak gamme,
    given the probability of winning a point.
    """

    def __call__(self, params):
        p = np.asarray(params[0])
        q = 1 - p
        t = []
        t.append(p ** 7)
        t.append(7 * p ** 7 * q)
        t.append(28 * p ** 7 * q ** 2)
        t.append(84 * p ** 7 * q ** 3)
        t.append(210 * p ** 7 * q ** 4)
        t.append(462 * p ** 7 * q ** 5)
        t.append(924 * p ** 8 * q ** 6 / (1 - 2 * p * q))
        return Tensor(np.stack(t).swapaxes(0, -1), dim=1)
    
    def grad(self, params, i):
        p = np.asarray(params[0])
        q = 1 - p
        d = []
        d.append(7 * p ** 6)
        d.append(7 * (7 - 8 * p) * p ** 6)
        d.append(28 * (7 - 9 * p) * q * p ** 6)
        d.append(84 * (7 - 10 * p) * q ** 2 * p ** 6)
        d.append(210 * (7 - 11 * p) * q ** 3 * p ** 6)
        d.append(462 * (7 - 12 * p) * q ** 4 * p ** 6)
        d.append(-1848 * (12 * p ** 3 - 19 * p ** 2 + 14 * p - 4) *
                 q ** 5 * p ** 7 / (1 - 2 * p * q) ** 2)
        return grad_tensor(np.stack(d).swapaxes(0, -1),
                           params, i, True, dim=1)
    
    def hess(self, params, i, j):
        p = np.asarray(params[0])
        q = 1 - p
        h = []
        h.append(42 * p ** 5)
        h.append(98 * (3 - 4 * p) * p ** 5)
        h.append(56 * (36 * p ** 2 - 56 * p + 21) * p ** 5)
        h.append(504 * (15 * p ** 2 - 21 * p + 7) * q * p ** 5)
        h.append(420 * (55 * p ** 2 - 70 * p + 21) * q ** 2 * p ** 5)
        h.append(924 * (66 * p ** 2 - 77 * p + 21) * q ** 3 * p ** 5)
        h.append(1848 * (264 * p ** 6 - 836 * p ** 5 + 1270 * p ** 4 -
                         1136 * p ** 3 + 625 * p ** 2 - 200 * p + 28) *
                 q ** 4 * p ** 6 / (1 - 2 * p * q) ** 3)
        return hess_tensor(np.stack(h).swapaxes(0, -1),
                           params, i, j, True, True, dim=1)


class Set(Func):
    """
    Func with probabilities of winning a (tennis) set,
    given the probabilities of winning a game and a tiebreak.
    """

    def __call__(self, params):
        p, t = params
        p = np.minimum(p, 1 - 1e-10)
        p = np.maximum(p, 1e-10)
        q = 1 - p
        s = []
        s.append(p ** 6)                    # s60
        s.append(6 * p ** 6 * q)            # s61
        s.append(21 * p ** 6 * q ** 2)      # s62
        s.append(56 * p ** 6 * q ** 3)      # s63
        s.append(126 * p ** 6 * q ** 4)     # s64
        s.append(252 * p ** 7 * q ** 5)     # s75
        s.append(504 * p ** 6 * q ** 6 * t) # s76
        return Tensor(np.stack(s).swapaxes(0, -1), dim=1)
    
    def grad(self, params, i):
        p, t = params
        q = 1 - p

        d = []
        if i == 0:
            d.append(6 * p ** 5)
            d.append(6 * (6 - 7 * p) * p ** 5)
            d.append(42 * (3 - 4 * p) * q * p ** 5)
            d.append(168 * (2 - 3 * p) * q ** 2 * p ** 5)
            d.append(252 * (3 - 5 * p) * q ** 3 * p ** 5)
            d.append(252 * (7 - 12 * p) * q ** 4 * p ** 6)
            d.append(3024 * (1 - 2 * p) * q ** 5 * p ** 5 * t)
        elif i == 1:
            z = np.zeros_like(p)
            d = 6 * [z]
            d.append(504 * q ** 6 * p ** 6)
        else:
            raise ValueError

        return grad_tensor(np.stack(d).swapaxes(0, -1),
                           params, i, True, dim=1)
    
    def hess(self, params, i, j):
        p, t = params
        q = 1 - p

        h = []
        if i == 0 and j == 0:
            h.append(30 * p ** 4)
            h.append(36 * (5 - 7 * p) * p ** 4)
            h.append(42 * (28 * p ** 2 - 42 * p + 15) * p ** 4)
            h.append(336 * (5 - 6 * p) * (1 - 2 * p) * q * p ** 4)
            h.append(756 * (15 * p ** 2 - 18 * p + 5) * q ** 2 * p ** 4)
            h.append(504 * (66 * p ** 2 - 77 * p + 21) * q ** 3 * p ** 5)
            h.append(3024 * (22 * p ** 2 - 22 * p + 5) * q ** 4 * p ** 4 * t)
        elif i == 1 and j == 0:
            z = np.zeros_like(p)
            h = 6 * [z]
            h.append(3024 * (1 - 2 * p) * q ** 5 * p ** 5)
        elif i == 1 and j == 1:
            z = np.zeros_like(p)
            h = 7 * [z]
        else:
            raise ValueError

        return hess_tensor(np.stack(h).swapaxes(0, -1),
                           params, i, j, True, True, dim=1)
