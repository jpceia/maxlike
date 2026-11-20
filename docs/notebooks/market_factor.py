"""
A generic model for K related ladders of time-to-event contracts (prediction
markets: "event happens by t_k" or "situation holds through t_k").

Each ladder k has a hazard curve; on the complementary log-log scale a
contract of ladder k observed at time t with deadline T is

    y = log H = log sum_j L_j exp(a_k[j]) + m_k(t) + eps,

    L_j: days of calendar bin j between t and T (bins: between the
         deadlines of the ladder), a_k: baseline log-hazards,
    m(t) in R^K: log-multipliers of the K hazard curves (level),
    and optionally n(t) in R^K, a "front" factor that moves the near end of
    each curve more than the far end:
        log h_k(s | t) = a_k(s) + m_k(t) + n_k(t) exp(-(s - t) / ell),
    so log H = log sum_j L_j exp(a_k[j] + n_k phi_j) + m_k;
    the state x = (m, n) is a random walk, x(t) = x(t-1) + eta,
    eta ~ N(0, Sigma) per time step (Sigma: 2K x 2K, within and across
    curves),
    eps: price noise of tau_k probability points, carried to that scale by
         the delta method: sd(eps) = tau_k / (H exp(-H)).

Given (a, Sigma, tau) the model is linear and Gaussian in the level m
(exact Kalman filter); the front factor enters non-linearly and is filtered
with an extended Kalman filter. The fair mid of a contract is its price on
the filtered curve, which uses the contracts of its own ladder and, through
Sigma, those of the others. It is always monotone in the deadline: the
cumulative hazard is a sum of positive terms whatever the state.
"""
import numpy as np
import pandas as pd
from polymarket_models import fit_dense

DAY = np.timedelta64(1, "D")


class Ladders:
    """
    data: DataFrame with columns ladder, deadline, time, price (one row per
    contract and time step); kinds: {ladder: "by" | "through"}.
    """

    def __init__(self, data, kinds, step="1h", min_hours=6, clip=2e-3,
                 front=None):
        data = data[(data["deadline"] - data["time"]) /
                    np.timedelta64(1, "h") > min_hours].copy()
        self.names = list(kinds)
        self.kinds = kinds
        self.K = len(self.names)
        start = data["time"].min().floor(step)
        self.grid = pd.date_range(start, data["time"].max().floor(step),
                                  freq=step)
        data["t"] = ((data["time"].dt.floor(step) - start) /
                     pd.Timedelta(step)).astype(int)
        data["k"] = data["ladder"].map({n: i for i, n in
                                        enumerate(self.names)})
        data = data.sort_values(["t", "k", "deadline"]).reset_index(drop=True)
        self.data = data
        q = np.clip(data["price"].values, clip, 1 - clip)
        self.q = q
        by = np.array([kinds[n] == "by" for n in self.names])[data["k"]]
        self.by = by
        H = np.where(by, -np.log1p(-q), -np.log(q))
        self.y = np.log(H)
        self.g = H * np.exp(-H)                  # |dp / d log H|
        self.k = data["k"].values
        self.t = data["t"].values
        # calendar bins per ladder and exposures
        self.edges, self.L = [], []
        tt = data["time"].dt.floor(step).values
        for k, n in enumerate(self.names):
            dl = np.sort(data.loc[data["k"] == k, "deadline"].unique())
            e = np.concatenate([[np.datetime64(start)], dl])
            self.edges.append(e)
        self.L = [self._exposure(tt, data["deadline"].values, k)
                  for k in range(self.K)]
        # front factor: phi = exp(-(days ahead of the middle of each
        # exposed part of a bin) / front)
        self.front = front
        self.F = 1 if front is None else 2
        if front is not None:
            self.phi = []
            for k in range(self.K):
                e = self.edges[k]
                lo = np.maximum(tt[:, None], e[None, :-1])
                hi = np.minimum(data["deadline"].values[:, None], e[None, 1:])
                mid = (lo + (hi - lo) / 2 - tt[:, None]) / DAY
                self.phi.append(np.exp(-np.maximum(mid, 0) / front))
        # observation indices per time step
        self.rows = np.split(np.arange(len(data)),
                             np.flatnonzero(np.diff(self.t)) + 1)
        self.steps = self.t[[r[0] for r in self.rows]]

    def _exposure(self, start, deadline, k):
        e = self.edges[k]
        lo = np.maximum(start[:, None], e[None, :-1])
        hi = np.minimum(deadline[:, None], e[None, 1:])
        return np.maximum((hi - lo) / DAY, 0.)

    # ------------------------------------------------------------ baselines
    def log_c(self, a, n=None, rows=None, jac=False):
        """
        log sum_j L_j exp(a_k[j] + n_k phi_j) for the observations rows
        (all by default), n: front state per observation; with jac also its
        derivative in n (the exposure-weighted average of phi).
        """
        rows = np.arange(len(self.q)) if rows is None else rows
        out = np.empty(rows.size)
        d = np.zeros(rows.size)
        for k in range(self.K):
            sel = self.k[rows] == k
            if not sel.any():
                continue
            o = rows[sel]
            w = self.L[k][o] * np.exp(a[k])[None, :]
            if self.F == 2 and n is not None:
                w = w * np.exp(n[sel][:, None] * self.phi[k][o])
            C = w.sum(1)
            out[sel] = np.log(C)
            if jac and self.F == 2:
                d[sel] = (w * self.phi[k][o]).sum(1) / C
        return (out, d) if jac else out

    def fit_baselines(self, x_hat, tau, a0, lam=1.):
        """
        Weighted least squares of y - m_hat on log C(a) (weights from the
        noise model), with a smoothness penalty, fitted with maxlike.
        """
        a = []
        for k in range(self.K):
            o = self.k == k
            L = self.L[k][o]
            if self.F == 2:
                L = L * np.exp(x_hat[self.K + k, self.t[o]][:, None] *
                               self.phi[k][o])
            r0 = self.y[o] - x_hat[k, self.t[o]]
            w = (self.g[o] / tau[k]) ** 2
            J = L.shape[1]
            Dm = np.diff(np.eye(J), axis=0)
            Q = lam * Dm.T @ Dm

            def fn(ak, L=L, r0=r0, w=w, Q=Q):
                CL = L * np.exp(ak)[None, :]
                C = CL.sum(1)
                pi = CL / C[:, None]
                r = r0 - np.log(C)
                v = -.5 * np.sum(w * r * r) - ak @ Q @ ak
                g = pi.T @ (w * r) - 2 * Q @ ak
                wr = w * r
                H = -(pi.T * w) @ pi + np.diag(pi.T @ wr) - (pi.T * wr) @ pi \
                    - 2 * Q
                return v, g, H

            mle = fit_dense(fn, [a0[k]])
            a.append(np.ma.getdata(mle.params[0]).astype(float))
        return a

    def start_baselines(self):
        a = []
        for k in range(self.K):
            o = self.k == k
            rate = np.exp(self.y[o]).sum() / self.L[k][o].sum()
            a.append(np.full(self.L[k].shape[1], np.log(rate)))
        return a

    # --------------------------------------------------------------- Kalman
    def kalman(self, a, Sigma, tau, smooth=False, P0=10.):
        """
        (Extended) Kalman filter, and optionally the RTS smoother. State
        x = (m_1..m_K[, n_1..n_K]). Returns the log-likelihood, the filtered
        means (D, T) and covariances, and the smoothed ones if asked. Time
        steps without observations only predict.
        """
        K, T = self.K, len(self.grid)
        D = K * self.F
        R = (np.asarray(tau)[self.k] / self.g) ** 2
        x, P = np.zeros(D), np.eye(D) * P0
        xf, Pf = np.zeros((D, T)), np.zeros((D, D, T))
        xp, Pp = np.zeros((D, T)), np.zeros((D, D, T))
        ll = 0.
        obs = dict(zip(self.steps, self.rows))
        for t in range(T):
            if t > 0:
                P = P + Sigma
            xp[:, t], Pp[:, :, t] = x, P
            rows = obs.get(t)
            if rows is not None:
                kk = self.k[rows]
                n = x[K + kk] if self.F == 2 else None
                hc, dn = self.log_c(a, n, rows, jac=True)
                Z = np.zeros((rows.size, D))
                Z[np.arange(rows.size), kk] = 1
                if self.F == 2:
                    Z[np.arange(rows.size), K + kk] = dn
                v = self.y[rows] - hc - x[kk]
                S = Z @ P @ Z.T + np.diag(R[rows])
                Lc = np.linalg.cholesky(S)
                Si_v = np.linalg.solve(Lc.T, np.linalg.solve(Lc, v))
                PZ = P @ Z.T
                x = x + PZ @ Si_v
                Si_PZt = np.linalg.solve(Lc.T, np.linalg.solve(Lc, PZ.T))
                P = P - PZ @ Si_PZt
                P = (P + P.T) / 2
                ll += -.5 * (v @ Si_v) - np.log(np.diag(Lc)).sum() - \
                    .5 * rows.size * np.log(2 * np.pi)
            xf[:, t], Pf[:, :, t] = x, P
        out = {"loglik": ll, "filtered": xf, "filtered_var": Pf}
        if smooth:
            xs, Ps = xf.copy(), Pf.copy()
            for t in range(T - 2, -1, -1):
                G = Pf[:, :, t] @ np.linalg.inv(Pp[:, :, t + 1])
                xs[:, t] = xf[:, t] + G @ (xs[:, t + 1] - xp[:, t + 1])
                Ps[:, :, t] = Pf[:, :, t] + G @ (Ps[:, :, t + 1] -
                                                 Pp[:, :, t + 1]) @ G.T
            out.update(smoothed=xs, smoothed_var=Ps)
        return out

    # ------------------------------------------------------ hyperparameters
    def unpack(self, x):
        K, Dm = self.K, self.K * self.F
        Lc = np.zeros((Dm, Dm))
        Lc[np.tril_indices(Dm)] = x[:Dm * (Dm + 1) // 2]
        Lc[np.diag_indices(Dm)] = np.exp(np.diag(Lc))
        tau = np.exp(x[Dm * (Dm + 1) // 2:])
        return Lc @ Lc.T, tau

    def pack(self, Sigma, tau):
        Dm = self.K * self.F
        Lc = np.linalg.cholesky(Sigma)
        Lc[np.diag_indices(Dm)] = np.log(np.diag(Lc))
        return np.concatenate([Lc[np.tril_indices(Dm)], np.log(tau)])

    def fit_hyper(self, a, x0):
        from scipy.optimize import minimize

        def f(x):
            Sigma, tau = self.unpack(x)
            try:
                return -self.kalman(a, Sigma, tau)["loglik"]
            except np.linalg.LinAlgError:
                return 1e12

        res = minimize(f, x0, method="L-BFGS-B")
        return res.x, -res.fun

    def fit(self, n_iter=4, lam=1.):
        """alternates the baselines (given the smoothed states) and the
        covariance and noise parameters (exact Kalman likelihood)"""
        a = self.start_baselines()
        Sigma = np.eye(self.K * self.F) * 0.02 ** 2
        tau = np.full(self.K, 0.02)
        x = self.pack(Sigma, tau)
        for _ in range(n_iter):
            Sigma, tau = self.unpack(x)
            ms = self.kalman(a, Sigma, tau, smooth=True)["smoothed"]
            # the level of m and of the baselines trade off: centre m
            for k in range(self.K):
                seen = np.unique(self.t[self.k == k])
                ms[k] -= ms[k, seen].mean()
                if self.F == 2:
                    ms[self.K + k] -= ms[self.K + k, seen].mean()
            a = self.fit_baselines(ms, tau, a, lam)
            x, ll = self.fit_hyper(a, x)
        self.a, self.x, self.loglik = a, x, ll
        self.Sigma, self.tau = self.unpack(x)
        return self

    def hyper_cov(self, h=1e-3):
        """covariance of the hyperparameters from a numerical Hessian"""
        x0, n = self.x, self.x.size

        def f(x):
            S, tau = self.unpack(x)
            return self.kalman(self.a, S, tau)["loglik"]

        H = np.zeros((n, n))
        f0 = f(x0)
        for i in range(n):
            for j in range(i, n):
                ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
                H[i, j] = H[j, i] = (f(x0 + ei + ej) - f(x0 + ei - ej) -
                                     f(x0 - ei + ej) + f(x0 - ei - ej)) / \
                    (4 * h * h)
        return np.linalg.inv(-H)

    def correlations(self, n_draws=4000, seed=0):
        """correlation matrix and 95% intervals (draws from the asymptotic
        normal law of the hyperparameters)"""
        cov = self.hyper_cov()
        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal(self.x, cov, n_draws)
        cors = []
        for d in draws:
            S, _ = self.unpack(d)
            s = np.sqrt(np.diag(S))
            cors.append(S / np.outer(s, s))
        cors = np.array(cors)
        S = self.Sigma
        s = np.sqrt(np.diag(S))
        return S / np.outer(s, s), np.quantile(cors, .025, 0), \
            np.quantile(cors, .975, 0)

    def regressions(self):
        """E[eta_k | eta_others] = B[k] eta_others, from Sigma"""
        S, K = self.Sigma, self.Sigma.shape[0]
        B = np.full((K, K), np.nan)
        for k in range(K):
            o = [j for j in range(K) if j != k]
            B[k, o] = S[k, o] @ np.linalg.inv(S[np.ix_(o, o)])
        return B

    # ------------------------------------------------------------- fair mid
    def price(self, xs, a=None):
        """
        price of every observed contract on the curve given by the states
        xs (D, T) at its time step: exact, so monotone in the deadline
        """
        a = self.a if a is None else a
        n = xs[self.K + self.k, self.t] if self.F == 2 else None
        H = np.exp(self.log_c(a, n) + xs[self.k, self.t])
        return np.where(self.by, -np.expm1(-H), np.exp(-H))

    def fair_mid(self, own_only=False):
        """
        Real-time fair mid of every observation: the filtered state at its
        time step (from the prices up to and including that step). With
        own_only each ladder is filtered from its own prices alone.
        """
        S = self.Sigma.copy()
        if own_only:
            # keep the covariance within each curve, drop it across curves
            lad = np.tile(np.arange(self.K), self.F)
            S[lad[:, None] != lad[None, :]] = 0
        xf = self.kalman(self.a, S, self.tau)["filtered"]
        return self.price(xf)
