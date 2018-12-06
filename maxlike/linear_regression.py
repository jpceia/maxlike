import abc
import numpy as np


class BaseLinearRegression(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, tol=1e-8, max_steps=20, verbose=False):
        self.tol = tol
        self.max_steps = max_steps
        self.verbose = verbose

    def y_pred(self, a, X):
        return (a * X).sum(1)

    def L(self, X, y, y_pred):
        raise NotImplementedError

    def grad(self, X, y, y_pred):
        raise NotImplementedError

    def hess(self, X, y, y_pred):
        raise NotImplementedError

    def param_guess(self, X, y):
        return np.linalg.solve(X.transpose().dot(X), X.transpose().dot(y))

    def fit(self, X, y):
        if X.ndim < 2:
            X = X.reshape(-1, 1)
        a = self.param_guess(X, y)
        y_pred = self.y_pred(a, X)
        L = self.L(X, y, y_pred) / X.shape[0]
        if self.verbose:
            print("step {}: L={}".format(0, -L))
        for k in range(self.max_steps):
            step = np.linalg.solve(
                self.hess(X, y, y_pred),
                self.grad(X, y, y_pred))
            m = 1
            for _ in range(50):
                a_new = a - m * step
                y_pred = self.y_pred(X, a_new)
                L_new = self.L(X, y, y_pred) / X.shape[0]
                if L_new >= L:
                    L = L_new
                    if self.verbose:
                        print("step {}: L={}".format(k + 1, -L))
                    break
                m *= .5
            else:
                raise ValueError
            if np.linalg.norm(a - a_new) < self.tol:
                break
            a = a_new
        else:
            raise ValueError
        self.a_ = a
        return self

    def predict(self, X):
        if X.ndim < 2:
            X = X.reshape(-1, 1)
        return self.y_pred(self.a_, X)


class PoissonLinearRegression(BaseLinearRegression):

    def y_pred(self, a, X):
        return (a * X).sum(1)

    def L(self, X, y, y_pred):
        return (y * np.log(y_pred) - y_pred).sum()

    def grad(self, X, y, y_pred):
        return ((1 - y / y_pred)[:, None] * X).sum(0)

    def hess(self, X, y, y_pred):
        return ((y / y_pred / y_pred)[:, None, None] *
                 X[:, None, :] * X[:, :, None]).sum(0)


class GaussianLinearRegression(BaseLinearRegression):

    def fit(self, X, y):
        self.a_ = self.param_guess(X, y)
        e = self.y_pred(self.a_, X) - y
        self.s_ = (e * e).sum() / (y.size - self.a_.size - 1)
        return self
