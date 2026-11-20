"""
Models for two related ladders of prediction-market contracts:

    A: "the US announces the end of the Iranian blockade by t_k"
       price ~ P(T_A <= t_k)
    B: "the Israel x Iran ceasefire continues through t_k"
       price ~ P(T_B > t_k)

Both are times to an event, described by a hazard rate. Three models:

1. FactorHistory: the price history. The hazard curve of each ladder,
   seen on day d, is a calendar baseline times a daily multiplier,
       h_X(t | d) = exp(a_X[bin(t)] + m_X[d]),
   with random-walk priors on m_A and m_B. The co-movement of m_A and m_B
   measures how the two events depend on each other.
2. FactorForward: today's prices with the multipliers evolving as a
   bivariate random walk from today (simulated paths): marginal prices
   are fitted exactly and the joint distribution of the two events
   follows from the paths.
3. MarkovForward: a four-state Markov chain (ceasefire on/broken x
   blockade on/ended) in which each event changes the hazard of the
   other by a factor (rho, kappa).

Every objective comes with its exact gradient and Hessian, and is fitted
with maxlike.LogLikelihood through DenseObjective.
"""
import numpy as np
from maxlike.tensor import Tensor
from maxlike.func.func_base import Func

DAY = np.timedelta64(1, "D")


# --------------------------------------------------------------------------
# objectives with dense derivatives, as maxlike Funcs
# --------------------------------------------------------------------------
class DenseObjective(Func):
    """
    A scalar objective fn(flat) -> (value, gradient, Hessian) of the
    concatenated parameters, as a Func with a single observation (fit it
    with maxlike.LogLikelihood and N = [1]).
    """

    def __init__(self, fn, shapes):
        self.fn = fn
        self.shapes = [tuple(np.shape(np.zeros(s))) for s in shapes]
        sizes = [int(np.prod(s)) for s in self.shapes]
        self.slices = np.split(np.arange(sum(sizes)), np.cumsum(sizes)[:-1])

    def flat(self, params):
        return np.concatenate([np.ravel(np.ma.getdata(p)).astype(float)
                               for p in params])

    def eval(self, params):
        v, g, H = self.fn(self.flat(params))
        blocks = list(zip(self.slices, self.shapes))
        grads = [Tensor(g[s].reshape(sh + (1,)), x_dim=len(sh))
                 for s, sh in blocks]
        hess = [[Tensor(H[np.ix_(si, sj)].reshape(shi + shj + (1,)),
                        x_dim=len(shi), y_dim=len(shj))
                 for sj, shj in blocks[:i + 1]]
                for i, (si, shi) in enumerate(blocks)]
        return Tensor(np.array([v])), grads, hess

    def __call__(self, params):
        return self.eval(params)[0]

    def grad(self, params, i):
        return self.eval(params)[1][i]

    def hess(self, params, i, j):
        return self.eval(params)[2][i][j]


def fit_dense(fn, values, fixed=None, names=None):
    """Fits a DenseObjective; values: list of starting arrays."""
    import maxlike
    mle = maxlike.LogLikelihood()
    mle.model = DenseObjective(fn, [np.shape(v) for v in values])
    for k, v in enumerate(values):
        mle.add_param(v, None if fixed is None else fixed[k],
                      name=None if names is None else names[k])
    mle.fit(N=np.ones(1), check_identifiability=False)
    return mle


def binomial_kl(q, p, dp, d2p, w=1.):
    """
    sum_k w_k [q_k log p_k + (1 - q_k) log(1 - p_k)] and its derivatives,
    given p (K,), dp (K, P) and d2p (K, P, P).
    """
    w = np.broadcast_to(w, q.shape)
    v = np.sum(w * (q * np.log(p) + (1 - q) * np.log1p(-p)))
    l1 = w * (q / p - (1 - q) / (1 - p))
    l2 = w * (-q / p ** 2 - (1 - q) / (1 - p) ** 2)
    g = dp.T @ l1
    H = np.einsum("k,ka,kb->ab", l2, dp, dp) + np.einsum("k,kab->ab", l1, d2p)
    return v, g, H


def exposure(start, deadline, edges):
    """Days of (start, deadline] in each bin (edges[j], edges[j+1]]."""
    lo = np.maximum(start[:, None], edges[None, :-1])
    hi = np.minimum(deadline[:, None], edges[None, 1:])
    return np.maximum((hi - lo) / DAY, 0.)


# --------------------------------------------------------------------------
# 1. factor model on the price history
# --------------------------------------------------------------------------
class FactorHistory:
    """
    Observation o: a price q of ladder s (0 = A, 1 = B) on day d for the
    deadline t. With C = sum_j L_j exp(a_s[j]) (L: days of (now, t] in
    each calendar bin) and H = C exp(m_s[d]) the cumulative hazard,
        A: p = 1 - exp(-H),     B: p = exp(-H).
    Objective: binomial KL to the prices, minus random-walk penalties
    lam_m sum (m[d] - m[d-1])^2 and smoothness penalties
    lam_a sum (a[j] - a[j-1])^2.
    """

    def __init__(self, data, edges, days, lam_m=5., lam_a=1., min_days=1.):
        data = data[(data["deadline"] - data["time"]) / DAY > min_days]
        self.data = data.reset_index(drop=True)
        self.edges = edges                          # per ladder
        self.days = days                            # day grid
        self.lam_m, self.lam_a = lam_m, lam_a
        self.ladder = (data["ladder"] == "ceasefire").values.astype(int)
        self.q = np.clip(data["price"].values, 1e-3, 1 - 1e-3)
        self.day = np.searchsorted(days, data["time"].values.astype(
            "datetime64[D]"))
        t, dl = data["time"].values, data["deadline"].values
        self.L = [exposure(t, dl, e) for e in edges]
        self.nb = [len(e) - 1 for e in edges]
        self.D = len(days)
        nA, nB, D = self.nb[0], self.nb[1], self.D
        self.sl = {"aA": slice(0, nA), "aB": slice(nA, nA + nB),
                   "mA": slice(nA + nB, nA + nB + D),
                   "mB": slice(nA + nB + D, nA + nB + 2 * D)}
        self.P = nA + nB + 2 * D
        # penalty matrix: - theta' Q theta
        Q = np.zeros((self.P, self.P))
        for key, lam in (("aA", lam_a), ("aB", lam_a), ("mA", lam_m),
                         ("mB", lam_m)):
            idx = np.arange(self.P)[self.sl[key]]
            Dm = np.diff(np.eye(idx.size), axis=0)
            Q[np.ix_(idx, idx)] += lam * Dm.T @ Dm
        self.Q = Q

    def objective(self, theta):
        n = self.q.size
        x = np.empty(n)
        J = np.zeros((n, self.P))
        Gw = np.zeros(n)
        pis = []
        for s, (a_key, m_key) in enumerate((("aA", "mA"), ("aB", "mB"))):
            o = self.ladder == s
            a, m = theta[self.sl[a_key]], theta[self.sl[m_key]]
            CL = self.L[s][o] * np.exp(a)[None, :]
            C = CL.sum(1)
            x[o] = np.log(C) + m[self.day[o]]
            pi = CL / C[:, None]
            J[np.ix_(np.flatnonzero(o), np.arange(self.P)[self.sl[a_key]])] = pi
            J[np.flatnonzero(o), self.sl[m_key].start + self.day[o]] = 1.
            pis.append((o, a_key, pi))
        H = np.exp(x)
        # ell = c1 log(1 - exp(-H)) - c0 H; A: c1 = q, B: c1 = 1 - q
        c1 = np.where(self.ladder == 0, self.q, 1 - self.q)
        c0 = 1 - c1
        e = -np.expm1(-H)                       # 1 - exp(-H)
        ell = c1 * np.log(e) - c0 * H
        with np.errstate(over="ignore"):        # large H: the terms go to 0
            G = H * (c1 / np.expm1(H) - c0)     # d ell / dx
            K = G - H * H * c1 / (np.expm1(H) * e)  # d2 ell / dx2
        v = ell.sum() - theta @ self.Q @ theta
        g = J.T @ G - 2 * self.Q @ theta
        Hs = J.T @ (K[:, None] * J) - 2 * self.Q
        for o, a_key, pi in pis:
            idx = np.arange(self.P)[self.sl[a_key]]
            Go = G[o]
            Hs[np.ix_(idx, idx)] += np.diag(pi.T @ Go) - pi.T @ (Go[:, None] * pi)
        return v, g, Hs

    def start(self):
        theta = np.zeros(self.P)
        for s, key in enumerate(("aA", "aB")):
            o = self.ladder == s
            q = self.q[o]
            H = -np.log1p(-q) if s == 0 else -np.log(q)
            rate = H.sum() / self.L[s][o].sum()
            theta[self.sl[key]] = np.log(rate)
        return theta

    def fit(self):
        theta = self.start()
        values = [theta[self.sl[k]] for k in ("aA", "aB", "mA", "mB")]
        fixed = [None, None, np.arange(self.D) == self.D - 1,
                 np.arange(self.D) == self.D - 1]
        self.mle = fit_dense(self.objective, values, fixed,
                             ["baseline A", "baseline B", "multiplier A",
                              "multiplier B"])
        self.theta = self.mle.model.flat(self.mle.params)
        return self

    def fitted_prices(self):
        theta = self.theta
        p = np.empty(self.q.size)
        for s, (a_key, m_key) in enumerate((("aA", "mA"), ("aB", "mB"))):
            o = self.ladder == s
            H = (self.L[s][o] * np.exp(theta[self.sl[a_key]])).sum(1) * \
                np.exp(theta[self.sl[m_key]][self.day[o]])
            p[o] = -np.expm1(-H) if s == 0 else np.exp(-H)
        return p

    def multipliers(self):
        return (self.theta[self.sl["mA"]], self.theta[self.sl["mB"]])


# --------------------------------------------------------------------------
# 2. forward factor model: bivariate random walk of the multipliers
# --------------------------------------------------------------------------
class FactorForward:
    """
    From today t0, the log-multipliers (W_A, W_B) follow a bivariate random
    walk with daily covariance Sigma, W(t0) = 0. Given a path,
        H_X(t) = sum_j exp(a_X[j]) int_{bin j, s <= t} exp(W_X(s)) ds,
    and the prices are averages over paths:
        A: p = E[1 - exp(-H_A)],   B: p = E[exp(-H_B)].
    The baselines a_A, a_B are fitted to today's prices (as many as there
    are contracts: an exact fit). The same paths give the joint law.
    """

    def __init__(self, t0, deadlines, Sigma, n_paths=4000, dt=0.25, seed=0):
        self.t0 = np.datetime64(t0)
        self.deadlines = [np.asarray(d, dtype="datetime64[ns]")
                          for d in deadlines]
        horizon = max((d.max() - self.t0) / DAY for d in self.deadlines)
        n_steps = int(np.ceil(horizon / dt))
        self.dt = dt
        # step mid-points (days after t0)
        self.s = (np.arange(n_steps) + .5) * dt
        rng = np.random.default_rng(seed)
        chol = np.linalg.cholesky(Sigma * dt)
        steps = rng.normal(size=(n_paths, n_steps, 2)) @ chol.T
        W = np.cumsum(steps, axis=1) - .5 * steps   # value at mid-points
        self.expW = np.exp(W)                       # (paths, steps, 2)
        # E[p, j, k]: integral over bin j up to deadline k, per ladder
        self.E = []
        for x, dl in enumerate(self.deadlines):
            tk = (dl - self.t0) / DAY
            edges = np.concatenate([[0.], tk])
            # share of each step in bin j (steps are fully inside, up to dt)
            inbin = ((self.s[:, None] > edges[None, :-1]) &
                     (self.s[:, None] <= edges[None, 1:])).astype(float) * dt
            upto = (self.s[:, None] <= tk[None, :]).astype(float)
            # (paths, steps) x (steps, j) and deadline mask
            E = np.einsum("ps,sj,sk->pjk", self.expW[:, :, x], inbin, upto)
            self.E.append(E)

    def _cum(self, x, a):
        """H (paths, K) and shares pi (paths, J, K)"""
        CL = self.E[x] * np.exp(a)[None, :, None]
        H = CL.sum(1)
        return H, CL / H[:, None, :]

    def marginal(self, x, a, derivatives=False):
        """prices of ladder x and their derivatives in a_x"""
        H, pi = self._cum(x, a)
        S = np.exp(-H)
        F = 1 - S if x == 0 else S
        p = F.mean(0)
        if not derivatives:
            return p
        # d S / d log H = -H S ; d2 = (-H + H^2) S
        dS, d2S = -H * S, (H * H - H) * S
        if x == 0:
            dS, d2S = -dS, -d2S
        dp = np.einsum("pk,pjk->kj", dS, pi) / H.shape[0]
        d2p = (np.einsum("pk,pjk,pik->kji", d2S, pi, pi) +
               np.einsum("pk,pjk->kj", dS, pi)[:, :, None] *
               np.eye(pi.shape[1])[None] -
               np.einsum("pk,pjk,pik->kji", dS, pi, pi)) / H.shape[0]
        return p, dp, d2p

    def fit(self, prices):
        self.prices = prices
        self.a = []
        for x in (0, 1):
            q = prices[x]
            K = q.size

            def fn(a, x=x, q=q):
                p, dp, d2p = self.marginal(x, a, True)
                return binomial_kl(q, p, dp, d2p)

            tk = (self.deadlines[x] - self.t0) / DAY
            H = -np.log1p(-q) if x == 0 else -np.log(q)
            rate = np.diff(np.concatenate([[0.], H])) / np.diff(
                np.concatenate([[0.], tk]))
            a0 = np.log(np.maximum(rate, 1e-4))
            mle = fit_dense(fn, [a0])
            self.a.append(np.ma.getdata(mle.params[0]).astype(float))
        return self

    def survival_paths(self, x):
        """P(event of ladder x has not happened by each deadline), per path"""
        H, _ = self._cum(x, self.a[x])
        return np.exp(-H)

    def joint(self, i, k):
        """P(blockade ended by its deadline i and ceasefire broken by k)"""
        SA, SB = self.survival_paths(0), self.survival_paths(1)
        return np.mean((1 - SA[:, i]) * (1 - SB[:, k]))


# --------------------------------------------------------------------------
# 3. four-state Markov chain
# --------------------------------------------------------------------------
# states: 0 = (ceasefire on, blockade on), 1 = (on, blockade ended),
#         2 = (ceasefire broken, blockade on), 3 = (broken, ended)
def _surv(x):
    """exp(-e^x) and its first two derivatives in x"""
    s = np.exp(-np.exp(x))
    return s, -np.exp(x) * s, (np.exp(2 * x) - np.exp(x)) * s


class MarkovForward:
    """
    Piecewise-constant hazards h_A (blockade ends) and h_B (ceasefire
    breaks), on the bins between today's deadlines of each ladder (the
    last bin of B extends to the end of A's horizon). Once the ceasefire
    has broken the blockade hazard is rho h_A; once the blockade has ended
    the ceasefire hazard is kappa h_B. Steps of dt days; within a step the
    two events are independent.

    Parameters (flat): a_A (J_A), a_B (J_B), log rho, log kappa.
    """

    def __init__(self, t0, deadlines, dt=0.25):
        self.t0 = np.datetime64(t0)
        self.tk = [(np.asarray(d, dtype="datetime64[ns]") - self.t0) / DAY
                   for d in deadlines]
        horizon = max(t.max() for t in self.tk)
        n = int(np.ceil(horizon / dt))
        self.dt = dt
        self.ends = (np.arange(n) + 1) * dt          # end of each step
        mid = self.ends - dt / 2
        self.bin = [np.minimum(np.searchsorted(t, mid), t.size - 1)
                    for t in self.tk]
        self.JA, self.JB = self.tk[0].size, self.tk[1].size
        self.P = self.JA + self.JB + 2
        # step index ending at each deadline (nearest)
        self.at = [np.clip(np.rint(t / dt).astype(int) - 1, 0, n - 1)
                   for t in self.tk]

    def _coef(self, kind, step):
        """x = c . theta + log dt for the hazards of a step"""
        c = np.zeros(self.P)
        if kind in ("A", "A2"):
            c[self.bin[0][step]] = 1
            if kind == "A2":
                c[self.P - 2] = 1
        else:
            c[self.JA + self.bin[1][step]] = 1
            if kind == "B1":
                c[self.P - 1] = 1
        return c

    def _factor(self, theta, kind, step, survive):
        c = self._coef(kind, step)
        s, ds, d2s = _surv(c @ theta + np.log(self.dt))
        if not survive:
            s, ds, d2s = 1 - s, -ds, -d2s
        return s, ds * c, d2s * np.outer(c, c)

    @staticmethod
    def _prod(f1, f2):
        v1, g1, h1 = f1
        v2, g2, h2 = f2
        return (v1 * v2, g1 * v2 + v1 * g2,
                h1 * v2 + np.outer(g1, g2) + np.outer(g2, g1) + v1 * h2)

    def _matrix(self, theta, step):
        P = self.P
        M = np.zeros((4, 4))
        dM = np.zeros((4, 4, P))
        d2M = np.zeros((4, 4, P, P))
        sA = lambda keep: self._factor(theta, "A", step, keep)
        sB = lambda keep: self._factor(theta, "B", step, keep)
        entries = {
            (0, 0): self._prod(sA(True), sB(True)),
            (0, 1): self._prod(sA(False), sB(True)),
            (0, 2): self._prod(sA(True), sB(False)),
            (0, 3): self._prod(sA(False), sB(False)),
            (1, 1): self._factor(theta, "B1", step, True),
            (1, 3): self._factor(theta, "B1", step, False),
            (2, 2): self._factor(theta, "A2", step, True),
            (2, 3): self._factor(theta, "A2", step, False),
        }
        for (i, j), (v, g, h) in entries.items():
            M[i, j], dM[i, j], d2M[i, j] = v, g, h
        M[3, 3] = 1
        return M, dM, d2M

    def propagate(self, theta, derivatives=True, start=None, first=0):
        """state probabilities after each step (and derivatives)"""
        P = self.P
        pi = np.array([1., 0, 0, 0]) if start is None else np.asarray(start)
        dpi = np.zeros((4, P))
        d2pi = np.zeros((4, P, P))
        out = []
        for step in range(first, self.ends.size):
            M, dM, d2M = self._matrix(theta, step)
            if derivatives:
                d2pi = (np.einsum("iab,ij->jab", d2pi, M) +
                        np.einsum("ia,ijb->jab", dpi, dM) +
                        np.einsum("ib,ija->jab", dpi, dM) +
                        np.einsum("i,ijab->jab", pi, d2M))
                dpi = (np.einsum("ia,ij->ja", dpi, M) +
                       np.einsum("i,ija->ja", pi, dM))
            pi = pi @ M
            out.append((pi, dpi, d2pi) if derivatives else pi)
        return out

    def marginals(self, theta, derivatives=True):
        """prices of the contracts: A (ended by t_k), B (holds through t_k)"""
        traj = self.propagate(theta, derivatives)
        res = []
        for x, states in ((0, [1, 3]), (1, [0, 1])):
            rows = [traj[s] for s in self.at[x]]
            if derivatives:
                p = np.array([r[0][states].sum() for r in rows])
                dp = np.array([r[1][states].sum(0) for r in rows])
                d2p = np.array([r[2][states].sum(0) for r in rows])
                res.append((p, dp, d2p))
            else:
                res.append(np.array([r[states].sum() for r in rows]))
        return res

    def fit(self, prices, rho=1., kappa=1.):
        qA, qB = prices

        def fn(theta):
            (pA, dA, d2A), (pB, dB, d2B) = self.marginals(theta)
            vA, gA, HA = binomial_kl(qA, pA, dA, d2A)
            vB, gB, HB = binomial_kl(qB, pB, dB, d2B)
            return vA + vB, gA + gB, HA + HB

        a0 = []
        for x, q in ((0, qA), (1, qB)):
            H = -np.log1p(-q) if x == 0 else -np.log(q)
            rate = np.diff(np.concatenate([[0.], H])) / np.diff(
                np.concatenate([[0.], self.tk[x]]))
            a0.append(np.log(np.maximum(rate, 1e-4)))
        mle = fit_dense(fn, [a0[0], a0[1], np.log(rho), np.log(kappa)],
                        [None, None, True, True])
        self.theta = mle.model.flat(mle.params)
        self.mle = mle
        return self

    def conditional_A(self, s_index):
        """
        P(blockade ended by each A deadline after the B deadline s_index |
        ceasefire broken by then), and the same given it held.
        """
        traj = self.propagate(self.theta, derivatives=False)
        k = self.at[1][s_index]
        pi = traj[k]
        out = {}
        for name, keep in (("broken", [2, 3]), ("holds", [0, 1])):
            start = np.zeros(4)
            start[keep] = pi[keep]
            mass = start.sum()
            later = self.propagate(self.theta, False, start, k + 1)
            probs = []
            for t_step in self.at[0]:
                if t_step < k:          # needs the path before s: not given
                    probs.append(np.nan)
                    continue
                state = start if t_step == k else later[t_step - k - 1]
                probs.append(state[[1, 3]].sum() / mass)
            out[name] = np.array(probs)
        return out


def factor_conditional_A(model, s_index):
    """
    Factor model: P(blockade ended by each A deadline | ceasefire broken by
    the B deadline s_index), and given that it held.
    """
    SA, SB = model.survival_paths(0), model.survival_paths(1)
    broken = 1 - SB[:, s_index]
    endA = 1 - SA
    return {"broken": (endA * broken[:, None]).mean(0) / broken.mean(),
            "holds": (endA * (1 - broken)[:, None]).mean(0) /
            (1 - broken).mean()}


def factor_conditional_B(model, a_index, b_index):
    """Factor model: P(ceasefire holds through B deadline b_index | blockade
    ended by A deadline a_index)."""
    SA, SB = model.survival_paths(0), model.survival_paths(1)
    ended = 1 - SA[:, a_index]
    return (SB[:, b_index] * ended).mean() / ended.mean()


def markov_conditional_B(model, a_index, b_index):
    """Markov: P(ceasefire holds through B deadline b_index | blockade ended
    by the earlier A deadline a_index)."""
    traj = model.propagate(model.theta, derivatives=False)
    k = model.at[0][a_index]
    start = np.zeros(4)
    start[[1, 3]] = traj[k][[1, 3]]
    mass = start.sum()
    later = model.propagate(model.theta, False, start, k + 1)
    t_step = model.at[1][b_index]
    state = start if t_step == k else later[t_step - k - 1]
    return state[1] / mass


# --------------------------------------------------------------------------
# 1b. the factor covariance as a parameter: Gaussian state-space model
# --------------------------------------------------------------------------
class FactorStateSpace:
    """
    Observation o (ladder s, day d): on the complementary log-log scale,
        r_o = log H_o(market) - log C_o(a) = m_s(d) + eps_o,
    with the baselines a fixed (from FactorHistory) and noise of standard
    deviation tau in price units, carried to that scale by the delta
    method: sd(eps_o) = tau / (H_o exp(-H_o)).

    State: (m_A, m_B) is a bivariate random walk anchored at today
    (m(today) = 0), with steps N(0, Sigma),
        Sigma = [[sA^2, rho sA sB], [rho sA sB, sB^2]].

    Integrating the states out, r ~ N(0, Z K Z' + R): the exact marginal
    likelihood of (sA, sB, rho, tau), so the correlation rho is estimated
    as a parameter, with the price noise kept apart.
    """

    def __init__(self, history, max_sd=1.5):
        h = history
        q = h.q
        H = np.where(h.ladder == 0, -np.log1p(-q), -np.log(q))
        C = np.empty(q.size)
        for s, key in enumerate(("aA", "aB")):
            o = h.ladder == s
            C[o] = (h.L[s][o] * np.exp(h.theta[h.sl[key]])).sum(1)
        self.r = np.log(H) - np.log(C)
        # delta method: d p / d log H = H exp(-H) (in absolute value)
        self.g = H * np.exp(-H)
        D = h.D
        # m_s(d) = - sum of the steps after d (anchored at the last day)
        steps_after = np.maximum(D - 1 - h.day, 0)
        # covariance of the states of two observations: number of common
        # steps times Sigma[s, s']
        self.common = np.minimum(steps_after[:, None], steps_after[None, :])
        self.ladder = h.ladder
        self.day, self.D = h.day, D
        self.logC = np.log(C)
        self.q = q
        self.n = q.size

    def loglik(self, sA, sB, rho, tau):
        S = np.array([[sA * sA, rho * sA * sB], [rho * sA * sB, sB * sB]])
        V = self.common * S[self.ladder[:, None], self.ladder[None, :]]
        V[np.diag_indices(self.n)] += (tau / self.g) ** 2
        try:
            Lc = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return -np.inf
        z = np.linalg.solve(Lc, self.r)
        return -0.5 * (z @ z) - np.log(np.diag(Lc)).sum() - \
            0.5 * self.n * np.log(2 * np.pi)

    @staticmethod
    def _unpack(x):
        return np.exp(x[0]), np.exp(x[1]), np.tanh(x[2]), np.exp(x[3])

    def fit(self, rho=None):
        """maximum likelihood (rho fixed if given: for a profile)"""
        from scipy.optimize import minimize
        x0 = np.array([np.log(.1), np.log(.1), 0., np.log(.02)])
        if rho is None:
            f = lambda x: -self.loglik(*self._unpack(x))
            res = minimize(f, x0, method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-4,
                                    "maxiter": 4000})
            return dict(zip(("sA", "sB", "rho", "tau"),
                            self._unpack(res.x)), loglik=-res.fun)
        idx = [0, 1, 3]
        f = lambda y: -self.loglik(np.exp(y[0]), np.exp(y[1]), rho,
                                   np.exp(y[2]))
        res = minimize(f, x0[idx], method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 4000})
        return dict(sA=np.exp(res.x[0]), sB=np.exp(res.x[1]), rho=rho,
                    tau=np.exp(res.x[2]), loglik=-res.fun)

    # ---------------------------------------------------------------- states
    def _cov(self, est):
        sA, sB, rho, tau = est["sA"], est["sB"], est["rho"], est["tau"]
        return np.array([[sA * sA, rho * sA * sB], [rho * sA * sB, sB * sB]]), tau

    def posterior(self, est, days, upto=None):
        """
        Mean and standard deviation of (m_A, m_B) on the given day indices,
        given the observations of the days <= upto (all if None): the
        smoothed states, or the filtered ones in real time.
        """
        S, tau = self._cov(est)
        o = np.ones(self.n, bool) if upto is None else self.day <= upto
        D = self.D
        after_obs = np.maximum(D - 1 - self.day[o], 0)
        lad = self.ladder[o]
        V = np.minimum(after_obs[:, None], after_obs[None, :]) * \
            S[lad[:, None], lad[None, :]]
        V[np.diag_indices(o.sum())] += (tau / self.g[o]) ** 2
        days = np.asarray(days)
        after_day = np.maximum(D - 1 - days, 0)
        mean, sd = [], []
        for s in (0, 1):
            C = np.minimum(after_day[:, None], after_obs[None, :]) * S[s, lad][None, :]
            sol = np.linalg.solve(V, np.column_stack([self.r[o], C.T]))
            mean.append(C @ sol[:, 0])
            var = after_day * S[s, s] - np.einsum("ij,ji->i", C, sol[:, 1:])
            sd.append(np.sqrt(np.maximum(var, 0)))
        return np.array(mean), np.array(sd)

    def filtered(self, est):
        """real-time states: each day from the observations up to that day"""
        mean = np.zeros((2, self.D))
        for d in range(self.D):
            if np.any(self.day <= d):
                m, _ = self.posterior(est, [d], upto=d)
                mean[:, d] = m[:, 0]
        return mean
