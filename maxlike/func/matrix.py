from .func_base import Func, grad_tensor, hess_tensor, null_func
from ..tensor import Tensor
from array import array
import numpy as np


class CollapseMatrix(Func):
    """
    CollapseMatrix(conditions)(M)[k] = sum of M[x, y] over the cells where
        sign(A_k * x + B_k * y + C_k) == s_k

    Turns a joint distribution on its last two distribution axes (e.g. the
    score matrix of a match, home goals x away goals) into the
    probabilities of a list of events, on one new distribution axis. The
    default events are home win, draw and away win (sign(x - y) = 1, 0,
    -1); (1, 1, -2.5, 1) would be "over 2.5 goals". Linear, so its Hessian
    is zero.

    Parameters
    ----------
    conditions : tuple (A, B, C, s), or a list of them (optional)
        The events, in order.
    """

    hess = null_func

    def __init__(self, conditions=None):
        """
        Condition or list of conditions with the form
        sgn(A*x + B*y + C) == s
        """
        if conditions is None:
            # home win, draw, away win
            conditions = [
                (1, -1, 0, 1),
                (1, -1, 0, 0),
                (1, -1, 0, -1),
            ]
        elif isinstance(conditions, tuple):
            conditions = [conditions]
        self.conditions = list(conditions)

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
        x_dim = ones.ndim
        val = np.stack(val, -1)
        val = val.swapaxes(0, x_dim - 2)
        val = val.swapaxes(1, x_dim - 1)
        x_map = array('b', range(x_dim - 2))
        x_map.append(-1)
        x_map.append(-1)
        idx = tuple([None] * (x_dim - 2) + [Ellipsis])
        return Tensor(val[idx], x_dim=x_dim, dim=1, x_map=x_map)


def _markov_chain(s1, s2, steps, gen1, gen2, start, derivatives=True):
    """
    Distribution of a two-counter Markov chain after 'steps' steps, and
    its first and second derivatives with respect to s1 and s2.

    Each step is the linear map L(Z) = Z + s1 * gen1(Z) + s2 * gen2(Z),
    where gen1 / gen2 are the generators of an event of each team (moving
    probability mass one state, the last state being absorbing). The
    derivatives follow from differentiating the recursion:

        D1'  = L(D1) + gen1(M)            D2'  = L(D2) + gen2(M)
        H11' = L(H11) + 2 gen1(D1)        H22' = L(H22) + 2 gen2(D2)
        H12' = L(H12) + gen1(D2) + gen2(D1)

    All six arrays are stacked so each step is a few vectorised operations.
    """
    s1 = s1[(...,) + (None,) * (start.ndim)]
    s2 = s2[(...,) + (None,) * (start.ndim)]
    shape = np.broadcast_shapes(s1.shape[:-start.ndim], s2.shape[:-start.ndim])
    if not derivatives:
        M = np.broadcast_to(start, shape + start.shape).copy()
        flow, G1, G2 = (np.empty_like(M) for _ in range(3))
        for _ in range(steps):
            gen1(M, flow, G1)
            gen2(M, flow, G2)
            G1 *= s1
            G2 *= s2
            M += G1
            M += G2
        return M

    # stack: M, D1, D2, H11, H12, H22 (buffers reused across steps)
    Z = np.zeros((6, ) + shape + start.shape)
    Z[0] = start
    flow, G1, G2 = (np.empty_like(Z) for _ in range(3))
    for _ in range(steps):
        gen1(Z, flow, G1)
        gen2(Z, flow, G2)
        # source terms use the generators of M, D1, D2 at the step start
        src3 = 2 * G1[1]
        src4 = G1[2] + G2[1]
        src5 = 2 * G2[2]
        src1 = G1[0].copy()
        src2 = G2[0].copy()
        G1 *= s1
        G2 *= s2
        Z += G1
        Z += G2
        Z[1] += src1
        Z[2] += src2
        Z[3] += src3
        Z[4] += src4
        Z[5] += src5
    return Z


class MarkovMatrix(Func):
    """
    Joint distribution of the scores (x, y) of two teams after 'steps'
    periods, each team scoring with probability s1 = params[0] / steps
    (resp. s2) per period, scaled by skew[x, y] (resp. skew[y, x]). Scores
    are capped at size - 1.
    """

    def __init__(self, size=8, steps=18, skew=None):
        self.size = size
        self.steps = steps
        if skew is None:
            skew = np.ones((size, size))
        self.skew = skew

    def _generators(self):
        F1 = np.asarray(self.skew, dtype=float)
        F2 = F1.T

        def gen1(Z, flow, out):
            # team 1 scores: mass moves from (x, y) to (min(x + 1, cap), y)
            np.multiply(F1, Z, out=flow)
            np.negative(flow, out=out)
            out[..., 1:, :] += flow[..., :-1, :]
            out[..., -1, :] += flow[..., -1, :]
            return out

        def gen2(Z, flow, out):
            np.multiply(F2, Z, out=flow)
            np.negative(flow, out=out)
            out[..., :, 1:] += flow[..., :, :-1]
            out[..., :, -1] += flow[..., :, -1]
            return out

        return gen1, gen2

    def _start(self):
        start = np.zeros((self.size, self.size))
        start[0, 0] = 1.0
        return start

    def _rates(self, params):
        s1 = np.asarray(params[0] / self.steps)
        s2 = np.asarray(params[1] / self.steps)
        assert s1.shape == s2.shape
        assert np.all(s1 > 0)
        assert np.all(s2 > 0)
        assert np.all(s1 + s2 < 1)
        return s1, s2

    def __call__(self, params):
        s1, s2 = self._rates(params)
        gen1, gen2 = self._generators()
        M = _markov_chain(s1, s2, self.steps, gen1, gen2, self._start(),
                          derivatives=False)
        return Tensor(M, dim=2)

    def grad(self, params, i):
        return self.eval(params)[1][i]

    def hess(self, params, i, j):
        return self.eval(params)[2][i][j]

    def eval(self, params):
        s1, s2 = self._rates(params)
        gen1, gen2 = self._generators()
        M, D1, D2, H11, H12, H22 = _markov_chain(
            s1, s2, self.steps, gen1, gen2, self._start())
        return _markov_tensors(self.steps, params, 2, M, D1, D2, H11, H12, H22)


def _markov_tensors(steps, params, dim, M, D1, D2, H11, H12, H22):
    # derivatives were taken with respect to s = param / steps
    D1, D2 = D1 / steps, D2 / steps
    k = steps * steps
    H11, H12, H22 = H11 / k, H12 / k, H22 / k
    return Tensor(M, dim=dim), \
        [grad_tensor(D1, params, 0, True, dim=dim),
         grad_tensor(D2, params, 1, True, dim=dim)], \
        [[hess_tensor(H11, params, 0, 0, True, True, dim=dim)],
         [hess_tensor(H12, params, 0, 1, True, True, dim=dim),
          hess_tensor(H22, params, 1, 1, True, True, dim=dim)]]


class MarkovVector(MarkovMatrix):
    """
    Distribution of the goal difference after 'steps' periods (states
    -(size - 1) .. size - 1, the extremes absorbing), each team scoring
    with probability s1 = params[0] / steps (resp. s2) per period, scaled
    by skew[x] (resp. skew[2 * size - 2 - x]) in state x.
    """

    def __init__(self, size=8, steps=18, skew=None):
        assert size > 0
        self.size = size
        self.steps = steps
        if skew is None:
            skew = np.ones((2 * size - 1,))
        self.skew = skew

    def _generators(self):
        F1 = np.asarray(self.skew, dtype=float)
        F2 = F1[::-1]

        def gen1(Z, flow, out):
            # team 1 scores: the difference moves up (capped)
            np.multiply(F1, Z, out=flow)
            np.negative(flow, out=out)
            out[..., 1:] += flow[..., :-1]
            out[..., -1] += flow[..., -1]
            return out

        def gen2(Z, flow, out):
            np.multiply(F2, Z, out=flow)
            np.negative(flow, out=out)
            out[..., :-1] += flow[..., 1:]
            out[..., 0] += flow[..., 0]
            return out

        return gen1, gen2

    def _start(self):
        start = np.zeros(2 * self.size - 1)
        start[self.size - 1] = 1.0
        return start

    def __call__(self, params):
        s1, s2 = self._rates(params)
        gen1, gen2 = self._generators()
        M = _markov_chain(s1, s2, self.steps, gen1, gen2, self._start(),
                          derivatives=False)
        return Tensor(M, dim=1)

    # FuncMeta only wraps grad/hess defined in the class body itself
    def grad(self, params, i):
        return self.eval(params)[1][i]

    def hess(self, params, i, j):
        return self.eval(params)[2][i][j]

    def eval(self, params):
        s1, s2 = self._rates(params)
        gen1, gen2 = self._generators()
        M, D1, D2, H11, H12, H22 = _markov_chain(
            s1, s2, self.steps, gen1, gen2, self._start())
        return _markov_tensors(self.steps, params, 1, M, D1, D2, H11, H12, H22)
