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

    def like(self, params, y):
        return (self.X * y - self.N * np.exp(y) -
                gammaln(self.X + 1)).sum()

    def grad_like(self, params, y, der):
        delta = self.X - self.N * np.exp(y)
        return [(d * delta).sum() for d in der]

    def hess_like(self, params, y, der, hess):
        delta2 = self.N * np.exp(y)
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

    def like(self, params, y):
        return (self.X * y + self.Z * np.log(self.s) -
                self.N * np.log(np.exp(np.exp(y)) + self.s - 1) - 
                gammaln(self.X + 1)).sum()

    def grad_like(self, params, y, der):
        exp_y = np.exp(y)
        delta = self.X - self.N * exp_y / (1 + (self.s - 1) * np.exp(-exp_y))
        return [(d * delta).sum() for d in der]

    def hess_like(self, params, y, der, hess):
        exp_y = np.exp(y)
        t = exp_y / (1 + (self.s - 1) * np.exp(-exp_y))
        delta = self.X - self.N * t
        delta2 = self.N * t * (1 + t * (self.s - 1) * np.exp(-exp_y))
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
        return -(self.N * np.log(1 + np.exp(-y)) +
                 (self.N - self.X) * y).sum()

    def grad_like(self, params, y, der):
        # (X - p * N) * d_y
        delta = self.X - self.N / (1 + np.exp(-y))
        return [(d * delta).sum() for d in der]

    def hess_like(self, params, y, der, hess):
        # (X - p * N) * hess_y - p * (1 - p) * N * di_y * dj_y^T
        p = 1 / (1 + np.exp(-y))
        delta = self.X - p * self.N
        delta2 = p * (1 - p) * self.N
        return [[(hess[i][j] * delta -
                  der[i] * der[j].transpose() * delta2).sum()
                 for j in range(i + 1)]
                for i in range(len(der))]


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
        z = y.sum(False)
        return (self.N * (np.log(y) - np.log(z))).sum()

    def grad_like(self, params, y, der):
        """
        Derivative of
            sum_k sum_u N_{k, u} * (log p(x=u|k) - log Z(k))

        = sum_k sum_u N_{k, u} * ((d_i p) / p - (d_i (sum_k p)) / (sum_k p))
        """
        grad = []
        z = y.sum(False)
        for d in der:
            dz = d.sum(False)
            grad.append((self.N * (d / y - dz / z)).sum())
        return grad

    def hess_like(self, params, y, der, hess):
        H = []
        z = y.sum(False)
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
        return (gammaln(self.X + r) - gammaln(self.X + 1) +
                r * np.log(r) - gammaln(r) +
                self.X * y - (self.X + r * self.N) *
                np.log(r + np.exp(y))).sum()

    def grad_like(self, params, y, der):
        # grad_m f = x / m - (x + r) / (m + r)
        # sum => X / m - (X + r * N) / (m + r)
        m = np.exp(y)
        r = self.scale
        # delta = m *  grad_m f
        delta = self.X - (self.X + r * self.N) * m / (m + r)
        return [(d * delta).sum() for d in der]

    def hess_like(self, params, y, der, hess):
        # hess_m = (x + r) / (m + r)^2 - x / m^2
        # sum => (X + r * N) / (m + r)^2 - X / m^2
        m = np.exp(y)
        r = self.scale
        s = m / (m + r)
        delta = self.X - (self.X + r * self.N) * s
        delta2 = (self.X + r * self.N) * s * s - self.X + delta
        return [[(hess[i][j] * delta +
                  der[i] * der[j].transpose() * delta2).sum()
                 for j in range(i + 1)]
                for i in range(len(der))]
