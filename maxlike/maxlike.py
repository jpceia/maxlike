from .maxlike_base import *
from scipy.special import factorial


class Poisson(MaxLike):
    """
    Class to model data under a Poisson Regression.
    """

    def __init__(self):
        MaxLike.__init__(self)
        self.X = None

    def like(self, params):
        ln_y = self.model(params)
        return (self.X * ln_y - self.N * np.exp(ln_y) -
                np.log(factorial(self.X))).sum()

    def grad_like(self, params):
        delta = self.X - self.N * np.exp(self.model(params))
        return [(d * delta).sum() for d in self.model.grad(params)]

    def hess_like(self, params):
        y = self.N * np.exp(self.model(params))
        delta = self.X - y
        der = self.model.grad(params)
        return [[(self.model.hess(params, i, j) * delta -
                  der[i] * der[j].transpose() * y).sum()
                 for j in range(i + 1)] for i in range(len(der))]


class Logistic(MaxLike):
    """
    Class to model under a Logistic Regression
    """

    def __init__(self):
        MaxLike.__init__(self)
        self.X = None

    def like(self, params):
        """
        p = 1 / (1 + e^-y)
        like = sum_k x_k * ln p + (1 - x_k) * ln (1-p)
             = - ((N - X) * y + N * ln(1 + e^-y))
        """
        y = self.model(params)
        return -(self.N * np.log(1 + np.exp(-y)) + (self.N - self.X) * y).sum()

    def grad_like(self, params):
        # (X - p * N) * d_y
        delta = self.X - self.N / (1 + np.exp(-self.model(params)))
        return [(d * delta).sum() for d in self.model.grad(params)]

    def hess_like(self, params):
        # (X - p * N) * hess_y - p * (1 - p) * N * di_y * dj_y^T
        p = 1 / (1 + np.exp(-self.model(params)))
        der = self.model.grad(params)
        delta = self.X - p * self.N
        delta2 = p * (1 - p) * self.N
        return[[(self.model.hess(params, i, j) * delta -
                 der[i] * der[j].transpose() * delta2).sum()
                for j in range(i + 1)]
               for i in range(len(der))]


class Finite(MaxLike):
    """
    Class to model an probabilistic regression under an arbitrary
    Discrete Finite Distribution

    This class doesn't require the model function to be normalized,
    I.e, the probabilistic model doesnt needs to satisfy sum_i p(i) = 1
    However we require p(i) > 0 for every i

    Can be used to minimize the Kullback-Leibler divergence, replacing
    frequencies by probabilities.
    """

    def __init__(self, dim=1):
        self.dim = dim
        MaxLike.__init__(self)

    def like(self, params):
        """
        evaluation of:
            sum_k sum_u N_{k, u} * (log p(x=u|k) - log Z(k))
        where:
            p(x=u|k) unnormalized probability function, conditional to k
            Z(k) := sum_u p(x=u|k)
        """
        p = self.model(params)
        z = p.sum(False)
        return (self.N * (np.log(p) - np.log(z))).sum()

    def grad_like(self, params):
        """
        Derivative of
            sum_k sum_u N_{k, u} * (log p(x=u|k) - log Z(k))

        = sum_k sum_u N_{k, u} * ((d_i p) / p - (d_i (sum_k p)) / (sum_k p))
        """
        grad = []
        p = self.model(params)
        z = p.sum(False)
        for d in self.model.grad(params):
            dz = d.sum(False)
            grad.append((self.N * (d / p - dz / z)).sum())
        return grad

    def hess_like(self, params):
        hess = []
        p = self.model(params)
        z = p.sum(False)
        der = self.model.grad(params)
        dz = [d.sum(False) for d in der]
        for i in range(len(params)):
            hess_line = []
            for j in range(i + 1):
                h = self.model.hess(params, i, j)
                hz = h.sum(False)
                H1 = (h - der[i] * der[j].transpose() / p) / p
                H2 = (hz - dz[i] * dz[j].transpose() / z) / z
                hess_line.append((self.N * (H1 - H2)).sum())
            hess.append(hess_line)
        return hess


class NegativeBinomial(MaxLike):
    """
    Class to model an probabilistic regression under an arbitrary
    Negative Binomial Distribution
    """

    def __init__(self, scale=1):
        MaxLike.__init__(self)
        self.scale = scale  # the model uses a fixed scale param
        self.X = None

    def like(self, params):
        # m = exp(y)
        # f = x * ln m - (x + r) * ln (r + m)
        # sum => X * ln m - (X + r * N) * ln (r + m)
        y = self.model(params)
        r = self.scale
        return (self.X * y - (self.X + r * self.N) *
                np.log(r + np.exp(y))).sum()

    def grad_like(self, params):
        # grad_m f = x / m - (x + r) / (m + r)
        # sum => X / m - (X + r * N) / (m + r)
        m = np.exp(self.model(params))
        r = self.scale
        # delta = m *  grad_m f
        delta = self.X - (self.X + r * self.N) * m / (m + r)
        return [(d * delta).sum() for d in self.model.grad(params)]

    def hess_like(self, params):
        # hess_m = (x + r) / (m + r)^2 - x / m^2
        # sum => (X + r * N) / (m + r)^2 - X / m^2
        m = np.exp(self.model(params))
        r = self.scale
        der = self.model.grad(params)
        s = m / (m + r)
        delta = self.X - (self.X + r * self.N) * s
        delta2 = (self.X + r * self.N) * s * s - self.X + delta
        return [[(self.model.hess(params, i, j) * delta +
                  der[i] * der[j].transpose() * delta2).sum()
                 for j in range(i + 1)]
                for i in range(len(der))]
