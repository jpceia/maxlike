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
                 front=None, front_ladders=None):
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
        # state: the K levels, then a front factor for each ladder in
        # front_ladders (all by default): fidx[k] is its index, or -1
        with_front = [] if front is None else (
            self.names if front_ladders is None else list(front_ladders))
        self.fidx = np.full(self.K, -1)
        for j, name in enumerate(with_front):
            self.fidx[self.names.index(name)] = self.K + j
        self.D = self.K + len(with_front)
        # ladder of each state dimension
        self.state_ladder = np.concatenate([
            np.arange(self.K), [self.names.index(n) for n in with_front]
        ]).astype(int)
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

    def _front(self, x, kk):
        """front state of each row (0 for ladders without one); x: (D,) or
        (D, n) indexed with columns matching kk"""
        f = self.fidx[kk]
        has = f >= 0
        if x.ndim == 1:
            vals = x[np.maximum(f, 0)]
        else:
            vals = x[np.maximum(f, 0), np.arange(kk.size)]
        return np.where(has, vals, 0.), has

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
            if self.fidx[k] >= 0:
                L = L * np.exp(x_hat[self.fidx[k], self.t[o]][:, None] *
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
        D = self.D
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
                n, has = self._front(x, kk)
                hc, dn = self.log_c(a, n, rows, jac=True)
                Z = np.zeros((rows.size, D))
                Z[np.arange(rows.size), kk] = 1
                r_f = np.flatnonzero(has)
                Z[r_f, self.fidx[kk[r_f]]] = dn[r_f]
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
            # lag-one covariances Cov(x_t, x_{t-1} | all), for EM
            lag = np.zeros((D, D, T))
            for t in range(T - 2, -1, -1):
                G = Pf[:, :, t] @ np.linalg.inv(Pp[:, :, t + 1])
                xs[:, t] = xf[:, t] + G @ (xs[:, t + 1] - xp[:, t + 1])
                Ps[:, :, t] = Pf[:, :, t] + G @ (Ps[:, :, t + 1] -
                                                 Pp[:, :, t + 1]) @ G.T
                lag[:, :, t + 1] = Ps[:, :, t + 1] @ G.T
            out.update(smoothed=xs, smoothed_var=Ps, lag_cov=lag)
        return out

    def em_step(self, a, Sigma, tau):
        """
        One EM step for (Sigma, tau): closed-form updates from the smoothed
        moments (for the front factor, around the smoothed states).
        """
        out = self.kalman(a, Sigma, tau, smooth=True)
        xs, Ps, lag = out["smoothed"], out["smoothed_var"], out["lag_cov"]
        T = xs.shape[1]
        dx = np.diff(xs, axis=1)
        Snew = (dx @ dx.T + Ps[:, :, 1:].sum(2) + Ps[:, :, :-1].sum(2) -
                lag[:, :, 1:].sum(2) - lag[:, :, 1:].sum(2).T) / (T - 1)
        Snew = (Snew + Snew.T) / 2
        # noise: E[(y - h(x))^2] = (y - h(x_s))^2 + z' P_s z
        K = self.K
        n, has = self._front(xs[:, self.t], self.k)
        hc, dn = self.log_c(a, n, jac=True)
        r = self.y - hc - xs[self.k, self.t]
        var = Ps[self.k, self.k, self.t]
        f = np.maximum(self.fidx[self.k], 0)
        var = var + has * (2 * dn * Ps[self.k, f, self.t] +
                           dn * dn * Ps[f, f, self.t])
        e2 = (r * r + var) * self.g ** 2
        tnew = np.sqrt(np.array([e2[self.k == k].mean()
                                 for k in range(K)]))
        return Snew, tnew, out["loglik"], xs

    def fit_em(self, n_iter=60, baseline_every=10, lam=1., tol=1e-3,
               verbose=False):
        """
        EM for Sigma and tau, refitting the baselines (with maxlike) every
        few steps. Each step is one pass of the smoother.
        """
        a = self.start_baselines()
        D = self.D
        Sigma = np.eye(D) * 0.02 ** 2
        tau = np.full(self.K, 0.02)
        self.trace = []
        for it in range(n_iter):
            Sigma, tau, ll, xs = self.em_step(a, Sigma, tau)
            self.trace.append(ll)
            if verbose:
                print(it, round(ll, 2))
            if it % baseline_every == 0:
                xs = xs.copy()
                for k in range(self.K):
                    seen = np.unique(self.t[self.k == k])
                    xs[k] -= xs[k, seen].mean()
                    if self.fidx[k] >= 0:
                        f = self.fidx[k]
                        xs[f] -= xs[f, seen].mean()
                a = self.fit_baselines(xs, tau, a, lam)
            elif len(self.trace) > 2 and \
                    abs(self.trace[-1] - self.trace[-2]) < tol:
                break
        self.a, self.Sigma, self.tau = a, Sigma, tau
        self.x = self.pack(Sigma, tau)
        self.loglik = self.kalman(a, Sigma, tau)["loglik"]
        return self

    # ------------------------------------------------------ hyperparameters
    def unpack(self, x):
        K, Dm = self.K, self.D
        Lc = np.zeros((Dm, Dm))
        Lc[np.tril_indices(Dm)] = x[:Dm * (Dm + 1) // 2]
        Lc[np.diag_indices(Dm)] = np.exp(np.diag(Lc))
        tau = np.exp(x[Dm * (Dm + 1) // 2:])
        return Lc @ Lc.T, tau

    def pack(self, Sigma, tau):
        Dm = self.D
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
        Sigma = np.eye(self.D) * 0.02 ** 2
        tau = np.full(self.K, 0.02)
        x = self.pack(Sigma, tau)
        for _ in range(n_iter):
            Sigma, tau = self.unpack(x)
            ms = self.kalman(a, Sigma, tau, smooth=True)["smoothed"]
            # the level of m and of the baselines trade off: centre m
            for k in range(self.K):
                seen = np.unique(self.t[self.k == k])
                ms[k] -= ms[k, seen].mean()
                if self.fidx[k] >= 0:
                    f = self.fidx[k]
                    ms[f] -= ms[f, seen].mean()
            a = self.fit_baselines(ms, tau, a, lam)
            x, ll = self.fit_hyper(a, x)
        self.a, self.x, self.loglik = a, x, ll
        self.Sigma, self.tau = self.unpack(x)
        return self

    # ------------------------------------------- exact scores of the filter
    def _dtheta(self, x):
        """dSigma / dtheta (n, D, D) and dlog(tau)/dtheta per ladder"""
        D, K = self.D, self.K
        Lc = np.zeros((D, D))
        Lc[np.tril_indices(D)] = x[:D * (D + 1) // 2]
        Lc[np.diag_indices(D)] = np.exp(np.diag(Lc))
        n = x.size
        dS = np.zeros((n, D, D))
        for idx, (r, c) in enumerate(zip(*np.tril_indices(D))):
            dL = np.zeros((D, D))
            dL[r, c] = Lc[r, c] if r == c else 1.
            dS[idx] = dL @ Lc.T + Lc @ dL.T
        dlt = np.zeros((n, K))
        dlt[D * (D + 1) // 2 + np.arange(K), np.arange(K)] = 1
        return dS, dlt

    def scores(self, x=None, a=None):
        """
        Exact contribution of every time step to the gradient of the
        log-likelihood in theta = (Cholesky factor of Sigma, log tau),
        from a tangent filter: dx and dP carried through the (extended)
        Kalman recursions. Returns (loglik, scores (T_obs, n)).
        """
        x = self.x if x is None else x
        a = self.a if a is None else a
        Sigma, tau = self.unpack(x)
        dSig, dlt = self._dtheta(x)
        n, D, K, T = x.size, self.D, self.K, len(self.grid)
        R_all = (np.asarray(tau)[self.k] / self.g) ** 2
        m, P = np.zeros(D), np.eye(D) * 10.
        dm, dP = np.zeros((n, D)), np.zeros((n, D, D))
        obs = dict(zip(self.steps, self.rows))
        ll, out = 0., []
        for t in range(T):
            if t > 0:
                P = P + Sigma
                dP = dP + dSig
            rows = obs.get(t)
            if rows is None:
                continue
            kk, nr = self.k[rows], rows.size
            nst, has = self._front(m, kk)
            hc, dn = self.log_c(a, nst, rows, jac=True)
            Z = np.zeros((nr, D))
            Z[np.arange(nr), kk] = 1
            dZ = np.zeros((n, nr, D))
            r_f = np.flatnonzero(has)
            if r_f.size:
                fc = self.fidx[kk[r_f]]
                Z[r_f, fc] = dn[r_f]
                vn = self._phi_var(a, nst, rows)
                dZ[:, r_f, fc] = vn[None, r_f] * dm[:, fc]
            R = R_all[rows]
            dR = 2 * dlt[:, kk] * R[None, :]
            v = self.y[rows] - hc - m[kk]
            dv = -np.einsum("rd,nd->nr", Z, dm)
            S = Z @ P @ Z.T + np.diag(R)
            dS = (np.einsum("nrd,de,se->nrs", dZ, P, Z) +
                  np.einsum("rd,nde,se->nrs", Z, dP, Z) +
                  np.einsum("rd,de,nse->nrs", Z, P, dZ))
            dS[:, np.arange(nr), np.arange(nr)] += dR
            Si = np.linalg.inv(S)
            Siv = Si @ v
            sign, logdet = np.linalg.slogdet(S)
            ll += -.5 * v @ Siv - .5 * logdet - .5 * nr * np.log(2 * np.pi)
            score = (-np.einsum("nr,r->n", dv, Siv) +
                     .5 * np.einsum("r,nrs,s->n", Siv, dS, Siv) -
                     .5 * np.einsum("rs,nsr->n", Si, dS))
            out.append(score)
            # update and its derivatives
            PZt = P @ Z.T
            Kg = PZt @ Si
            dPZt = np.einsum("nde,re->ndr", dP, Z) + \
                np.einsum("de,nre->ndr", P, dZ)
            dK = np.einsum("ndr,rs->nds", dPZt, Si) - \
                np.einsum("dr,nrs,st->ndt", Kg, dS, Si)
            m_new = m + Kg @ v
            dm = dm + np.einsum("ndr,r->nd", dK, v) + \
                np.einsum("dr,nr->nd", Kg, dv)
            P_new = P - Kg @ S @ Kg.T
            dP = (dP - np.einsum("ndr,rs,es->nde", dK, S, Kg) -
                  np.einsum("dr,nrs,es->nde", Kg, dS, Kg) -
                  np.einsum("dr,rs,nes->nde", Kg, S, dK))
            m, P = m_new, (P_new + P_new.T) / 2
            dP = (dP + dP.transpose(0, 2, 1)) / 2
        return ll, np.array(out)

    def _phi_var(self, a, n, rows):
        """d^2 log C / dn^2: the exposure-weighted variance of phi"""
        out = np.zeros(rows.size)
        for k in range(self.K):
            sel = self.k[rows] == k
            if not sel.any():
                continue
            o = rows[sel]
            w = self.L[k][o] * np.exp(a[k])[None, :] * \
                np.exp(n[sel][:, None] * self.phi[k][o])
            pi = w / w.sum(1, keepdims=True)
            ph = self.phi[k][o]
            mean = (pi * ph).sum(1)
            out[sel] = (pi * ph * ph).sum(1) - mean ** 2
        return out

    def refine(self, n_steps=10, tol=1e-6):
        """
        Scoring (BHHH) steps on (Sigma, tau): theta += I^-1 g, with the
        exact gradient g and the information I = sum_t s_t s_t'.
        """
        x = self.x
        ll, s = self.scores(x)
        for _ in range(n_steps):
            g, info = s.sum(0), s.T @ s
            # Levenberg damping: robust to weakly identified directions
            step = np.linalg.solve(info + 1e-6 * np.trace(info) / info.shape[0]
                                   * np.eye(info.shape[0]), g)
            lam = 1.
            while lam > 1e-4:
                ll_new, s_new = self.scores(x + lam * step)
                if ll_new >= ll - 1e-9:
                    break
                lam /= 2
            done = abs(ll_new - ll) < tol
            x, ll, s = x + lam * step, ll_new, s_new
            if done:
                break
        self.x, self.loglik = x, ll
        self.Sigma, self.tau = self.unpack(x)
        return self

    def hyper_cov(self):
        """covariance of the hyperparameters: inverse of the BHHH
        information from the exact scores (no finite differences)"""
        _, s = self.scores()
        return np.linalg.inv(s.T @ s)

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
        n, _ = self._front(xs[:, self.t], self.k)
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
            lad = self.state_ladder
            S[lad[:, None] != lad[None, :]] = 0
        xf = self.kalman(self.a, S, self.tau)["filtered"]
        return self.price(xf)
