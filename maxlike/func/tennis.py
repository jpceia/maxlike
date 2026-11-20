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
        return Tensor(np.stack(g, axis=-1), dim=1)
    
    def grad(self, params, i):
        p = np.asarray(params[0])
        q = 1 - p
        d = []
        d.append(4 * p ** 3)
        d.append(4 * (4 - 5 * p) * p ** 3)
        d.append(20 * (2 - 3 * p) * q * p ** 3)
        d.append(-20 * (12 * p ** 3 - 20 * p ** 2 + 16 * p - 5) *
                 q ** 2 * p ** 4 / (1 - 2 * p * q) ** 2)
        return grad_tensor(np.stack(d, axis=-1),
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
        return hess_tensor(np.stack(h, axis=-1),
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
        return Tensor(np.stack(t, axis=-1), dim=1)
    
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
        return grad_tensor(np.stack(d, axis=-1),
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
        return hess_tensor(np.stack(h, axis=-1),
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
        return Tensor(np.stack(s, axis=-1), dim=1)
    
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

        return grad_tensor(np.stack(d, axis=-1),
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

        return hess_tensor(np.stack(h, axis=-1),
                           params, i, j, True, True, dim=1)


class _Jet:
    """
    A value with its first and second derivatives in two variables (u, v):
    (f, f_u, f_v, f_uu, f_uv, f_vv), for exact derivatives through the
    recursions of ServeSet.
    """

    def __init__(self, f, fu=0., fv=0., fuu=0., fuv=0., fvv=0.):
        self.d = [f, fu, fv, fuu, fuv, fvv]

    @staticmethod
    def var(x, k):
        z = np.zeros_like(x)
        return _Jet(x, *(([1. + z, z] if k == 0 else [z, 1. + z]) +
                         [z, z, z]))

    def __add__(self, o):
        if not isinstance(o, _Jet):
            return _Jet(self.d[0] + o, *self.d[1:])
        return _Jet(*[a + b for a, b in zip(self.d, o.d)])

    __radd__ = __add__

    def __neg__(self):
        return _Jet(*[-a for a in self.d])

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, _Jet):
            return _Jet(*[a * o for a in self.d])
        f, fu, fv, fuu, fuv, fvv = self.d
        g, gu, gv, guu, guv, gvv = o.d
        return _Jet(f * g, fu * g + f * gu, fv * g + f * gv,
                    fuu * g + 2 * fu * gu + f * guu,
                    fuv * g + fu * gv + fv * gu + f * guv,
                    fvv * g + 2 * fv * gv + f * gvv)

    __rmul__ = __mul__

    def inv(self):
        g, gu, gv, guu, guv, gvv = self.d
        r = 1 / g
        return _Jet(r, -gu * r * r, -gv * r * r,
                    (2 * gu * gu / g - guu) * r * r,
                    (2 * gu * gv / g - guv) * r * r,
                    (2 * gv * gv / g - gvv) * r * r)

    def __truediv__(self, o):
        return self * o.inv() if isinstance(o, _Jet) else self * (1 / o)


def _game_jet(p):
    """probability of winning a game on a point probability p (a _Jet)"""
    q = 1 - p
    p2, q2 = p * p, q * q
    p4 = p2 * p2
    deuce = p2 / (1 - 2 * p * q)
    return p4 * (1 + 4 * q + 10 * q2) + 20 * (p2 * p) * (q2 * q) * deuce


def _tiebreak_jet(ps, pr, first_serves):
    """
    probability that player i wins a tie-break: i wins a point with ps on
    its serve and pr on the other's; i serves the first point if
    first_serves, then the serve alternates every two points
    """
    P = {(0, 0): 1.}
    win = 0.
    for total in range(12):                     # up to 6-6
        for x in range(total + 1):
            y = total - x
            if (x, y) not in P or x > 6 or y > 6:
                continue
            m = total + 1                       # number of the next point
            i_serves = ((m // 2) % 2 == 0) == first_serves
            w = ps if i_serves else pr
            prob = P.pop((x, y))
            nxt = [((x + 1, y), prob * w), ((x, y + 1), prob * (1 - w))]
            for (a, b), v in nxt:
                if a == 7 and b <= 5:
                    win = win + v
                elif b == 7 and a <= 5:
                    continue
                else:
                    P[(a, b)] = P.get((a, b), 0.) + v
    # from 6-6 each pair of points has one serve of each player
    both = ps * pr
    return win + P[(6, 6)] * both / (both + (1 - ps) * (1 - pr))


def _serve_set_jet(ps, pr):
    """
    probabilities that i wins the set 6-0, ..., 6-4, 7-5, 7-6, averaged
    over who serves the first game (1/2 each)
    """
    hold, brk = _game_jet(ps), _game_jet(pr)
    out = None
    for first in (True, False):
        P = {(0, 0): 1.}
        res = {}
        for total in range(12):
            for x in range(total + 1):
                y = total - x
                if (x, y) not in P:
                    continue
                n = total + 1                   # number of the next game
                i_serves = (n % 2 == 1) == first
                w = hold if i_serves else brk
                prob = P.pop((x, y))
                for (a, b), v in (((x + 1, y), prob * w),
                                  ((x, y + 1), prob * (1 - w))):
                    if (a == 6 and b <= 4) or (a == 7 and b == 5):
                        res[(a, b)] = v
                    elif (b == 6 and a <= 4) or (b == 7 and a == 5):
                        continue
                    else:
                        P[(a, b)] = P.get((a, b), 0.) + v
        # game 13 (the tie-break) is served by the first server
        tb = _tiebreak_jet(ps, pr, first)
        scores = [res[(6, k)] for k in range(5)] + [res[(7, 5)],
                                                    P[(6, 6)] * tb]
        scores = [s * .5 for s in scores]
        out = scores if out is None else [a + b for a, b in zip(out, scores)]
    return out


class ServeSet(Func):
    """
    Probabilities that a player wins a (tennis) set with each score (6-0,
    ..., 6-4, 7-5, 7-6), given the probabilities of winning a point on
    their own serve (ps) and on the opponent's serve (pr). Games alternate
    serve, the tie-break serve alternates every two points, and who serves
    first is unknown: the result is the average over the two cases.

    With ps = pr this is Set() @ [Collapse(Game(), False),
    Collapse(TieBreak(), False)].
    """

    def _eval(self, params):
        ps = np.clip(np.asarray(params[0], dtype=float), 1e-10, 1 - 1e-10)
        pr = np.clip(np.asarray(params[1], dtype=float), 1e-10, 1 - 1e-10)
        ps, pr = np.broadcast_arrays(ps, pr)
        jets = _serve_set_jet(_Jet.var(ps, 0), _Jet.var(pr, 1))
        return [np.stack([np.broadcast_to(j.d[k], ps.shape) for j in jets],
                         axis=-1) for k in range(6)]

    def __call__(self, params):
        return Tensor(self._eval(params)[0], dim=1)

    def grad(self, params, i):
        return grad_tensor(self._eval(params)[1 + i], params, i, True, dim=1)

    def hess(self, params, i, j):
        d = self._eval(params)
        k = {(0, 0): 3, (1, 0): 4, (0, 1): 4, (1, 1): 5}[(i, j)]
        return hess_tensor(d[k], params, i, j, True, True, dim=1)
