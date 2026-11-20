import numpy as np
from .maxlike_base import MaxLike
from scipy.special import gammaln


class Poisson(MaxLike):
    """
    Class to model data under a Poisson Regression.
    """

    def __init__(self):
        MaxLike.__init__(self)
        self.X = None

    def _rate(self, y):
        # N * exp(y), shared by like, grad_like and hess_like for the same y
        return self._memo("rate", y, lambda: self.N * np.exp(y))

    def _log_factorials(self):
        # sum(log X!) depends only on the data
        return self._memo("log_fact", self.X,
                          lambda: gammaln(self.X + 1).sum())

    def like(self, params, y):
        return (self.X * y - self._rate(y)).sum() - self._log_factorials()

    def grad_like(self, params, y, der):
        delta = self.X - self._rate(y)
        return [(d * delta).sum() for d in der]

    def hess_like(self, params, y, der, hess):
        delta2 = self._rate(y)
        delta = self.X - delta2
        return [[(hess[i][j] * delta -
                  der[i] * der[j].transpose() * delta2).sum()
                 for j in range(i + 1)] for i in range(len(der))]


class ZeroInflatedPoisson(MaxLike):
    """
    Class to model data under a Zero inflated Poisson distribution

        P(X=0)   = d / (e ** y + d - 1)
        P(X=x>0) = y ** x / (x! * (e ** y + d - 1))
    """
    def __init__(self, s=1):
        MaxLike.__init__(self)
        assert s >= 0
        self.s = s
        self.X = None
        self.Z = None

    def _terms(self, y):
        # exp(y), exp(-exp(y)) and t, shared by grad_like and hess_like
        def compute():
            exp_y = np.exp(y)
            e = np.exp(-exp_y)
            return exp_y, e, exp_y / (1 + (self.s - 1) * e)
        return self._memo("zip_terms", (y, self.s), compute)

    def _data_term(self):
        # sum(Z * log(s) - log X!) depends only on the data
        return self._memo(
            "zip_data", (self.X, self.Z, self.s),
            lambda: (self.Z * np.log(self.s) - gammaln(self.X + 1)).sum())

    def like(self, params, y):
        exp_y = self._terms(y)[0]
        return (self.X * y -
                self.N * np.log(np.exp(exp_y) + self.s - 1)).sum() + \
            self._data_term()

    def grad_like(self, params, y, der):
        t = self._terms(y)[2]
        delta = self.X - self.N * t
        return [(d * delta).sum() for d in der]

    def hess_like(self, params, y, der, hess):
        _, e, t = self._terms(y)
        delta = self.X - self.N * t
        delta2 = self.N * t * (1 + t * (self.s - 1) * e)
        return [[(hess[i][j] * delta - der[i] * der[j].transpose() * delta2).sum()
                 for j in range(i + 1)]
                for i in range(len(der))]


class Logistic(MaxLike):
    """
    Class to model under a Logistic Regression
    """

    def __init__(self):
        MaxLike.__init__(self)
        self.X = None

    def like(self, params, y):
        """
        p = 1 / (1 + e^-y)
        like = sum_k x_k * ln p + (1 - x_k) * ln (1-p)
             = - ((N - X) * y + N * ln(1 + e^-y))
        """
        return -(self.N * np.log(1 + self._exp_neg(y)) +
                 (self.N - self.X) * y).sum()

    def _exp_neg(self, y):
        # exp(-y), shared by like, grad_like and hess_like for the same y
        return self._memo("exp_neg", y, lambda: np.exp(-y))

    def grad_like(self, params, y, der):
        # (X - p * N) * d_y
        delta = self.X - self.N / (1 + self._exp_neg(y))
        return [(d * delta).sum() for d in der]

    def hess_like(self, params, y, der, hess):
        # (X - p * N) * hess_y - p * (1 - p) * N * di_y * dj_y^T
        p = 1 / (1 + self._exp_neg(y))
        delta = self.X - p * self.N
        delta2 = p * (1 - p) * self.N
        return [[(hess[i][j] * delta -
                  der[i] * der[j].transpose() * delta2).sum()
                 for j in range(i + 1)]
                for i in range(len(der))]


    def _post_fit_checks(self, params):
        # complete (or quasi-complete) separation: the maximum is at infinity
        # and some fitted probabilities collapse to 0 or 1
        import warnings
        y = self.model(params)
        y = np.asarray(y.toarray() if hasattr(y, "toarray") else y)
        N = np.asarray(getattr(self.N, "values", self.N))
        y, N = np.broadcast_arrays(y, N)
        p = 1 / (1 + np.exp(-y))
        observed = N > 0
        extreme = observed & ((p < 1e-8) | (p > 1 - 1e-8))
        if extreme.any():
            warnings.warn(
                "%d fitted probabilities are numerically 0 or 1: the data "
                "may be (quasi-)separated, so some estimates diverge and "
                "their standard errors are meaningless. Consider adding a "
                "regularization (add_regularization)." % extreme.sum(),
                RuntimeWarning)


class Finite(MaxLike):
    """
    Class to model an probabilistic regression under an arbitrary
    Discrete Finite Distribution

    This class doesn't require the model function to be normalized,
    I.e, the probabilistic model doesnt need to satisfy sum_i p(x=i) = 1
    However we require p(x=i) > 0 for every i

    Can be used to minimize the Kullback-Leibler divergence, replacing
    frequencies by probabilities.
    """

    def __init__(self, dim=1):
        self.dim = dim
        MaxLike.__init__(self)

    def like(self, params, y):
        """
        evaluation of:
            sum_k sum_u N_{k, u} * (log p(x=u|k) - log Z(k))
        where:
            p(x=u|k) unnormalized probability function, conditional to k
            Z(k) := sum_u p(x=u|k)
        """
        z = self._z(y)
        return (self.N * (np.log(y) - np.log(z))).sum()

    def _z(self, y):
        # normalising constant Z(k), shared by like, grad_like and hess_like
        return self._memo("z", y, lambda: y.sum(False))

    def grad_like(self, params, y, der):
        """
        Derivative of
            sum_k sum_u N_{k, u} * (log p(x=u|k) - log Z(k))

        = sum_k sum_u N_{k, u} * ((d_i p) / p - (d_i (sum_k p)) / (sum_k p))
        """
        grad = []
        z = self._z(y)
        for d in der:
            dz = d.sum(False)
            grad.append((self.N * (d / y - dz / z)).sum())
        return grad

    def hess_like(self, params, y, der, hess):
        H = []
        z = self._z(y)
        dz = [d.sum(False) for d in der]
        for i in range(len(params)):
            hess_line = []
            for j in range(i + 1):
                h = hess[i][j]
                hz = h.sum(False)
                H1 = (h - der[i] * der[j].transpose() / y) / y
                H2 = (hz - dz[i] * dz[j].transpose() / z) / z
                hess_line.append((self.N * (H1 - H2)).sum())
            H.append(hess_line)
        return H


class NormalizedFinite(MaxLike):
    """
    Class to model an probabilistic regression under an arbitrary
    Discrete Finite Distribution

    Can be used to minimize the Kullback-Leibler divergence, replacing
    frequencies by probabilities.
    """

    def __init__(self, dim=1):
        self.dim = dim
        MaxLike.__init__(self)

    def like(self, params, y):
        """
        evaluation of:
            sum_{k, u} N_{k, u} * log p(x=u|k)
        where:
            p(x=u|k) normalized probability function, conditional to k
        """
        return (self.N * np.log(y)).sum()

    def grad_like(self, params, y, der):
        return [(self.N * d / y).sum() for d in der]

    def hess_like(self, params, y, der, hess):
        H = []
        for i in range(len(params)):
            hess_line = []
            for j in range(i + 1):
                h = (hess[i][j] - der[i] * der[j].transpose() / y) / y
                hess_line.append((self.N * h).sum())
            H.append(hess_line)
        return H


class NegativeBinomial(MaxLike):
    """
    Class to model an probabilistic regression under an arbitrary
    Negative Binomial Distribution
    """

    def __init__(self, scale=1):
        MaxLike.__init__(self)
        self.scale = scale  # the model uses a fixed scale param
        self.X = None

    def like(self, params, y):
        # m = exp(y)
        # f = x * ln m - (x + r) * ln (r + m)
        # sum => X * ln m - (X + r * N) * ln (r + m)
        r = self.scale
        return (self.X * y - (self.X + r * self.N) *
                np.log(r + self._mean(y))).sum() + self._data_term()

    def _mean(self, y):
        # m = exp(y), shared by like, grad_like and hess_like for the same y
        return self._memo("mean", y, lambda: np.exp(y))

    def _data_term(self):
        # terms of the log-likelihood that depend only on the data
        r = self.scale
        return self._memo(
            "nb_data", (self.X, self.N, self.scale),
            lambda: (gammaln(self.X + r) - gammaln(self.X + 1) +
                     (r * np.log(r) - gammaln(r)) * (self.N * 0 + 1)).sum())

    def grad_like(self, params, y, der):
        # grad_m f = x / m - (x + r) / (m + r)
        # sum => X / m - (X + r * N) / (m + r)
        m = self._mean(y)
        r = self.scale
        # delta = m *  grad_m f
        delta = self.X - (self.X + r * self.N) * m / (m + r)
        return [(d * delta).sum() for d in der]

    def hess_like(self, params, y, der, hess):
        # hess_m = (x + r) / (m + r)^2 - x / m^2
        # sum => (X + r * N) / (m + r)^2 - X / m^2
        m = self._mean(y)
        r = self.scale
        s = m / (m + r)
        delta = self.X - (self.X + r * self.N) * s
        delta2 = (self.X + r * self.N) * s * s - self.X + delta
        return [[(hess[i][j] * delta +
                  der[i] * der[j].transpose() * delta2).sum()
                 for j in range(i + 1)]
                for i in range(len(der))]


class LogLikelihood(MaxLike):
    """
    Generic objective: the model returns the log-likelihood (or any score)
    of each observation, and the objective is its weighted sum,
        sum_t N[t] * y[t],
    with N the observation weights (e.g. ones). Use it for models whose
    likelihood is not one of the built-in distributions, e.g.
    maxlike.func.FirstOrderStatistic.
    """

    def __init__(self):
        MaxLike.__init__(self)

    def like(self, params, y):
        return (self.N * y).sum()

    def grad_like(self, params, y, der):
        return [(self.N * d).sum() for d in der]

    def hess_like(self, params, y, der, hess):
        return [[(self.N * hess[i][j]).sum() for j in range(i + 1)]
                for i in range(len(der))]
