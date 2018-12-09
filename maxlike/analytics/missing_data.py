import pandas as pd
import numpy as np
from pandas import isnull


def np_cov(X):
    return pd.DataFrame(X).cov().values


def _corr(S):
    std = np.sqrt(np.diag(S))
    return S / std[:, None] / std[None, :]


def __EM_step(u, S, X, X_mask):

    S_miss = np.zeros_like(S)

    # M step
    for miss in np.unique(X_mask, axis=0):
        if ~miss.any():
            continue
        filt = (X_mask == miss).all(1)
        z = X[filt][:, ~miss] - u[~miss]
        oo = np.ix_(~miss, ~miss)
        mo = np.ix_(miss, ~miss)
        om = np.ix_(~miss, miss)
        mm = np.ix_(miss, miss)
        S_inv = np.linalg.inv(S[oo])
        X[np.ix_(filt, miss)] = z.dot(S[mo].dot(S_inv).transpose()) + u[miss]
        q = filt.sum() * 1.0 / len(X)
        S_miss[mm] += q * (S[mm] - S[mo].dot(S_inv).dot(S[om]))

    # E step
    return np.nanmean(X, 0), np_cov(X) + S_miss, X


class GaussianImputer:

    def __init__(self, max_steps=100, tol=1e-8, verbose=True):
        self.max_steps = max_steps
        self.tol = tol
        self.verbose = verbose

    def fit(self, X):

        u = np.nanmean(X, 0)
        S = np_cov(X)
        X_mask = np.isnan(X)

        for t in range(self.max_steps):

            u0, S0 = u.copy(), S.copy()             # old values
            u, S, X = __EM_step(u0, S0, X, X_mask)  # new values

            # Measure error
            e = np.linalg.norm(S - S0) / np.linalg.norm(S0)

            # print message
            if self.verbose:
                print("t =", t, "\te = ", e)

            # stopping condition
            if e < self.tol:
                break
        else:
            raise ValueError("Did not converge after %d steps" %
                             self.max_steps)

        self.u_ = u
        self.S_ = S
        return self

    def _fill(self, X, u, S):
        X_mask = isnull(X)

        for miss in np.unique(X_mask, axis=0):
            if ~miss.any():
                continue

            filt = (X_mask == miss).all(1)
            z = X[filt][:, ~miss] - u[~miss]
            oo = np.ix_(~miss, ~miss)
            mo = np.ix_(miss, ~miss)
            X[np.ix_(filt, miss)] = S[mo].dot(
                np.linalg.solve(S[oo], z[..., None])) + u[miss]

        return X

    def transform(self, X):
        u = self.u_
        S = self.S_
        X = self._fill(X, u, S)
        return X
