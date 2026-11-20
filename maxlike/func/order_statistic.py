"""
First order statistic of conditionally independent latent values: the
smallest value "wins", and only the winner and the winning value are
observed.

Examples: auctions where the lowest bid wins (public procurement), races
where the fastest time wins, competing risks where the first event is
observed. For "largest value wins" pass the negated values.

Each participant j in each observation t draws a latent value

    v[t, j] = c[t] + mu[j] + sigma * eps[t, j],   eps ~ N(0, 1)

c[t] is an observation-level common factor (e.g. the cost of a tender) and
mu[j] the location of participant j (e.g. the log markup of a firm). Given
c[t] the values are independent, so the likelihood of "j = w wins with
value v" is

    f_w(v | c) * prod_{j participant, j != w} S_j(v | c)

(density of the winner's value times the probability that every other
participant drew a larger value). Two specifications of the common factor:

- independent given covariates: c[t] = x[t] . beta;
- with a latent common factor: c[t] = x[t] . beta + tau * u[t],
  u[t] ~ N(0, 1), integrated out with Gauss-Hermite quadrature. tau
  measures how much the values of an observation move together beyond the
  covariates.

FirstOrderStatistic returns the log-likelihood of each observation, to be
fitted with maxlike.LogLikelihood. See docs/first_order_statistic.md for
the formulas.
"""
import numpy as np
from scipy.special import log_ndtr
from ..tensor import Tensor
from .func_base import Func

_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def _mills(z):
    """phi(z) / Phi(-z), the hazard of the standard normal, computed stably."""
    return np.exp(-0.5 * z * z - _LOG_SQRT_2PI - log_ndtr(-z))


class FirstOrderStatistic(Func):
    """
    Log-likelihood of each observation (winner, winning value) when the
    smallest of the participants' latent values wins.

    Parameters of the Func, in order:
        mu         (n_participants,)  location of each participant
        log_sigma  scalar             log of the dispersion of the values
        beta       (n_cov,)           effect of the covariates on c[t]
        tau        scalar             weight of the latent common factor
                                      (only with common_factor=True; its
                                      sign is irrelevant)

    Data (fixed at construction):
        values       (n_obs,)                 winning (smallest) value
        winner       (n_obs,) int             index of the winner
        participants (n_obs, n_participants)  True for those who took part
        covariates   (n_obs, n_cov)           covariates of each observation
                                              (include a column of ones for
                                              an intercept)
    """

    def __init__(self, values, winner, participants, covariates,
                 common_factor=False, nodes=15):
        self.values = np.asarray(values, dtype=float)
        self.winner = np.asarray(winner, dtype=int)
        self.participants = np.asarray(participants, dtype=bool)
        self.covariates = np.asarray(covariates, dtype=float)
        n_obs, n_part = self.participants.shape
        assert self.values.shape == (n_obs,)
        assert self.covariates.shape[0] == n_obs
        assert np.all(self.participants[np.arange(n_obs), self.winner]), \
            "the winner must be among the participants"
        self.common_factor = common_factor
        win = np.zeros((n_obs, n_part), dtype=bool)
        win[np.arange(n_obs), self.winner] = True
        self._win = win
        self._lose = self.participants & ~win
        if common_factor:
            # probabilists' Gauss-Hermite: E[g(u)] ~ sum_k w_k g(u_k)
            u, w = np.polynomial.hermite_e.hermegauss(nodes)
            self._u, self._logw = u, np.log(w / w.sum())
        else:
            self._u, self._logw = np.zeros(1), np.zeros(1)

    def __call__(self, params):
        return self.eval(params)[0]

    def grad(self, params, i):
        return self.eval(params)[1][i]

    def hess(self, params, i, j):
        return self.eval(params)[2][i][j]

    def eval(self, params):
        mu = np.asarray(params[0], dtype=float)
        s = float(np.asarray(params[1]))
        beta = np.asarray(params[2], dtype=float)
        tau = float(np.asarray(params[3])) if self.common_factor else 0.
        sigma = np.exp(s)
        X = self.covariates                      # (T, Q)
        W = self._win[:, :, None]                # (T, J, 1)
        Ls = self._lose[:, :, None]
        u = self._u                              # (K,)

        # z[t, j, k] = (v[t] - c[t, k] - mu[j]) / sigma
        c = (X @ beta)[:, None] + tau * u[None, :]           # (T, K)
        z = (self.values[:, None, None] - c[:, None, :]
             - mu[None, :, None]) / sigma                     # (T, J, K)
        lam = _mills(z)
        # log-term of each participant and its first two derivatives in z:
        # winner: log phi(z) (density of the value, minus log sigma below),
        # others: log Phi(-z)
        ell_j = np.where(W, -0.5 * z * z - _LOG_SQRT_2PI,
                         np.where(Ls, log_ndtr(-z), 0.))
        g = np.where(W, -z, np.where(Ls, -lam, 0.))
        h = np.where(W, -1., np.where(Ls, -lam * (lam - z), 0.))
        ell = ell_j.sum(1) - s                                # (T, K)

        # sums over participants
        G = g.sum(1)                                          # (T, K)
        H = h.sum(1)
        A = (h * z + g).sum(1)                                # (T, K)
        B = (h * z * z + g * z).sum(1)

        inv = 1. / sigma
        # gradient of ell[t, k] per parameter block
        d_mu = -g * inv                                       # (T, J, K)
        d_s = -(g * z).sum(1) - 1.                            # (T, K)
        d_beta = -G[:, None, :] * X[:, :, None] * inv         # (T, Q, K)
        blocks = [d_mu, d_s, d_beta]
        if self.common_factor:
            d_tau = -G * u[None, :] * inv                     # (T, K)
            blocks.append(d_tau)

        # mixture over the quadrature nodes: log y = logsumexp(logw + ell)
        a = self._logw[None, :] + ell
        amax = a.max(1, keepdims=True)
        pi = np.exp(a - amax)
        tot = pi.sum(1, keepdims=True)
        logy = (amax + np.log(tot))[:, 0]
        pi = pi / tot                                         # (T, K)

        # expectation over nodes (weights pi) of each gradient block
        def mean(d):
            return np.einsum("t...k,tk->t...", d, pi)

        grads = [mean(d) for d in blocks]

        # Hessian of ell[t, k] blocks (x = block i, y = block j <= i)
        X2 = X * inv
        hess_ell = {
            (0, 0): None,   # diagonal in participants, handled below
            (1, 0): (h * z + g) * inv,                        # (T, J, K)
            (1, 1): B,
            (2, 0): h[:, None, :, :] * (X2 * inv)[:, :, None, None],
            (2, 1): A[:, None, :] * X2[:, :, None],           # (T, Q, K)
            (2, 2): H[:, None, None, :] * (X2[:, :, None] *
                                           X2[:, None, :])[..., None],
        }
        if self.common_factor:
            uu = u[None, :]
            hess_ell.update({
                (3, 0): h * uu[:, None, :] * inv * inv,      # (T, J, K)
                (3, 1): A * uu * inv,
                (3, 2): H[:, None, :] * X2[:, :, None] * uu[:, None, :] * inv,
                (3, 3): H * (uu * inv) ** 2,
            })

        # Hessian of log y = E[hess ell + d ell d ell^T] - E[d] E[d]^T
        n_blocks = len(blocks)
        hess = []
        for i in range(n_blocks):
            row = []
            for j in range(i + 1):
                di, dj = blocks[i], blocks[j]
                # E[d_i d_j^T]: outer product over the parameter axes
                ei = "t" + "abcd"[:di.ndim - 2] + "k"
                ej = "t" + "efgh"[:dj.ndim - 2] + "k"
                out = "t" + ei[1:-1] + ej[1:-1]
                second = np.einsum("%s,%s,tk->%s" % (ei, ej, out), di, dj, pi)
                if (i, j) == (0, 0):
                    # diagonal: d2 ell / dmu_m^2 = h_m / sigma^2
                    diag = np.einsum("tjk,tk->tj", h, pi) * inv * inv
                    idx = np.arange(diag.shape[1])
                    second[:, idx, idx] += diag
                elif (i, j) == (1, 0) or (i, j) == (3, 0):
                    second += np.einsum("tjk,tk->tj", hess_ell[(i, j)], pi)
                else:
                    second += np.einsum("t...k,tk->t...", hess_ell[(i, j)],
                                        pi)
                mi = grads[i].reshape(grads[i].shape[0], -1)
                mj = grads[j].reshape(grads[j].shape[0], -1)
                second -= np.einsum("ta,tb->tab", mi, mj).reshape(
                    second.shape)
                row.append(second)
            hess.append(row)

        # Tensors: parameter axes first, then the observation (feature) axis
        def tensor(arr, x_dim, y_dim=0):
            return Tensor(np.moveaxis(arr, 0, -1), x_dim=x_dim, y_dim=y_dim)

        ndims = [1, 0, 1, 0][:n_blocks]
        grad_t = [tensor(gr, ndims[i]) for i, gr in enumerate(grads)]
        hess_t = [[tensor(hess[i][j], ndims[i], ndims[j])
                   for j in range(i + 1)] for i in range(n_blocks)]
        return Tensor(logy), grad_t, hess_t
