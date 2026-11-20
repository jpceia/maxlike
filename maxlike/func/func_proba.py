import numpy as np
from scipy.special import factorial, gammaln
from ..tensor import Tensor
from .func import Exp
from .func_base import Func, grad_tensor, hess_tensor


class Poisson(Func):
    """
    Poisson distribution of a count x = 0, ..., size - 1 with mean lambda
    (params[0]), along a distribution axis.

    normalised=False (default): the unnormalised vector lambda^x / x!,
        without exp(-lambda). Enough for the Finite likelihood, which
        normalises over the distribution axis (the result is then the
        distribution truncated to x < size).
    normalised=True: probabilities. 'tail' sets what happens to the mass
        of x >= size:
        "censor"   (default) the last entry is P(X >= size - 1), e.g. for
                   counts recorded as "size - 1 or more"; sums to one;
        "truncate" the distribution conditional on X < size; sums to one;
        "drop"     the exact pmf P(X = x); sums to P(X < size).

    Derivatives use dP(x)/dlambda = P(x - 1) - P(x) and, for the censored
    tail, d/dlambda P(X >= k) = P(X = k - 1).
    """

    TAILS = ("censor", "truncate", "drop")

    def __init__(self, size=10, normalised=False, tail="censor"):
        if tail not in self.TAILS:
            raise ValueError("tail must be one of %s" % (self.TAILS, ))
        self.size = size
        self.normalised = normalised
        self.tail = tail

    def _vec(self, params):
        a = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        return (a ** x) / factorial(x)

    @staticmethod
    def _shift(vec):
        # d/da of (a ^ x) / x! is the same vector shifted by one position
        return np.insert(vec[..., :-1], 0, 0, -1)

    def _values(self, params):
        """value and its first two derivatives in lambda, along x"""
        if not self.normalised:
            u = self._vec(params)
            du = self._shift(u)
            return u, du, self._shift(du)

        lam = np.asarray(params[0], dtype=float)[..., None]
        x = np.arange(self.size)
        # exact pmf, computed on the log scale
        p = np.exp(x * np.log(lam) - lam - gammaln(x + 1))
        dp = self._shift(p) - p                  # P(x - 1) - P(x)
        d2p = self._shift(dp) - dp

        if self.tail == "drop":
            return p, dp, d2p

        if self.tail == "censor":
            # last entry: P(X >= size - 1) = 1 - P(X < size - 1)
            v, dv, d2v = p.copy(), dp.copy(), d2p.copy()
            v[..., -1] = 1 - p[..., :-1].sum(-1)
            dv[..., -1] = -dp[..., :-1].sum(-1)
            d2v[..., -1] = -d2p[..., :-1].sum(-1)
            return v, dv, d2v

        # truncate: v = p / Z, Z = P(X < size)
        Z = p.sum(-1, keepdims=True)
        dZ = dp.sum(-1, keepdims=True)
        d2Z = d2p.sum(-1, keepdims=True)
        v = p / Z
        dv = dp / Z - p * dZ / Z ** 2
        d2v = (d2p / Z - 2 * dp * dZ / Z ** 2 - p * d2Z / Z ** 2 +
               2 * p * dZ ** 2 / Z ** 3)
        return v, dv, d2v

    def __call__(self, params):
        return Tensor(self._values(params)[0], dim=1)

    def grad(self, params, i):
        return grad_tensor(self._values(params)[1], params, i, True, dim=1)

    def hess(self, params, i, j):
        return hess_tensor(self._values(params)[2], params, i, j, True, True,
                           dim=1)

    def eval(self, params):
        v, dv, d2v = self._values(params)
        return Tensor(v, dim=1), \
            [grad_tensor(dv, params, 0, True, dim=1)], \
            [[hess_tensor(d2v, params, 0, 0, True, True, dim=1)]]


class LogNegativeBinomial(Func):

    def __init__(self, size=10, r=1):
        self.size = size
        self.r = r

    def _val(self, params):
        """
        gammaln(r + x) - gammaln(r) - ln x! +
        x ln(m) + r ln(r) - (x + r) * ln(r + m))
        """
        m = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        vec = x * np.log(m)
        vec += gammaln(self.r + x)
        vec -= gammaln(self.r)
        vec += self.r * np.log(self.r)
        vec -= np.log(factorial(x))
        vec -= (x + self.r) * np.log(self.r + m)
        return vec

    def _grad(self, params):
        """
        grad_f = (x / m) - (x + r) / (r + m)
        """
        m = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        return x / m - (x + self.r) / (self.r + m)

    def _hess(self, params):
        """
        hess_f = (x + r) / (r + m) ** 2 - x / m ** 2
        """
        m = np.asarray(params[0])[..., None]
        x = np.arange(self.size)
        return (x + self.r) / np.square(self.r + m) - x / np.square(m)

    def __call__(self, params):
        return Tensor(self._val(params), dim=1)

    def grad(self, params, i):
        return grad_tensor(self._grad(params), params, i, True, dim=1)

    def hess(self, params, i, j):
        return hess_tensor(self._hess(params), params, i, j, True, True, dim=1)

    def eval(self, params):
        return Tensor(self._val(params), dim=1), \
               [grad_tensor(self._grad(params), params, 0, True, dim=1)], \
               [[hess_tensor(self._hess(params), params, 0, 0,
                             True, True, dim=1)]]


def NegativeBinomial(*args, **kwargs):
    return Exp() @ LogNegativeBinomial(*args, **kwargs)
