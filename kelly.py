import numpy as np


def bernoulli(odd, prob, prob_err=None):
    b = (odd * prob - 1) / (odd - 1)
    k = 1
    if prob_err is not None:
        s = (1 + 1 / odd) * prob_err
        k = b * b / (b * b + s * s)
    return k * b


def exclusive(o, p, dp_list=None):
    """
    Exclusive Kelly Algorithm:
    1.  Calculated expected revenues:
        E[r_i] = p_i * o_i
    2.  Reorder the indexes so that the sequence E[r_i] is nonincreasing
    3.  Set S = [], i = 1 and R = 1
    4.  Repeat
        if E[r_i] > R:
            insert i in S
            R := 1 - (sum_{not S} p_k) / (1 - sum_{S} (1 / o_i))
        else:
            break
    5.  f_i := p_i - (1 / o_i) * (sum_{not S} p_k) / (1 - sum_{S} (1 / o_i))
    """

    if o.ndim == 1 and p.ndim == 1:
        rev, q = p * o, 1 / o
        idx = np.argsort(-rev)
        tmp_p = 1 - np.cumsum(p[idx])
        tmp_q = 1 - np.cumsum(q[idx])
        R = np.insert(tmp_p / tmp_q, 0, 1, 0)[:-1]
        i = np.argmin(rev[idx] > R)
        b = np.maximum((rev - R[i]) * q, 0)
        k = 1

        if (b == 0).all():
            return b

        if dp_list is not None:

            if not isinstance(dp_list, (list, tuple)):
                dp_list = [dp_list]

            db_list = []
            for dp in dp_list:
                dR = np.insert(-np.cumsum(dp[idx]) / tmp_q, 0, 1, 0)[:-1]
                db = dp - dR[i] / o
                db[b == 0] = 0
                db_list.append(db)

            db = np.stack(db_list, -2)
            D = 1 - b.sum() + b * o
            M = np.diag(o) - 1
            H = -(M[None, :, :] * (p / D / D)[None, None, :] * M[:, None, :]).sum(-1)
            B = (db[:, None, :, None] * H[None, None, :, :] *
                 db[None, :, None, :]).sum((-1, -2))
            A = (b[:, None] * H * b[None, :]).sum((-1, -2))
            k = A / (A + B.sum((-1, -2)))

        return k * b

    elif o.ndim > 1 and p.ndim > 1:
        assert o.shape == p.shape
        rev, q = p * o, 1 / o
        idx = np.argsort(-rev, 1)
        rng = np.arange(p.shape[0])[:, None]
        tmp_p = 1 - np.cumsum(p[rng, idx], 1)
        tmp_q = 1 - np.cumsum(q[rng, idx], 1)
        R = np.insert(tmp_p / tmp_q, 0, 1, 1)[..., :-1]
        i = np.argmin(rev[rng, idx] > R, 1)[:, None]
        b = np.maximum((rev - R[rng, i]) * q, 0)
        k = np.array([1])
        if dp_list is not None:
            
            if not isinstance(dp_list, (list, tuple)):
                dp_list = [dp_list]

            db_list = []
            for dp in dp_list:
                tmp_dp = -np.cumsum(dp[rng, idx], 1)
                dR = np.insert(tmp_dp / tmp_q, 0, 1, 1)[..., :-1]
                db = dp - dR[rng, i] / o
                db[b == 0] = 0
                db_list.append(db)

            db = np.stack(db_list, -2)
            D = 1 - b.sum(-1)[:, None] + b * o
            M = o[:, None, :] * np.diag(np.ones(o.shape[-1])) - 1
            H = -(M[:, None, :] * (p / D / D)[:, None, None] * M[:, :, None]).sum(-1)
            B = (db[:, :, None, :, None] * H[:, None, None, :, :] *
                 db[:, None, :, None, :]).sum((-1, -2))
            A = (b[:, :, None] * H * b[:, None, :]).sum((-1, -2))
            k = A / (A + B.sum((-1, -2)))

        return k[:, None] * b

    elif o.ndim == 1 and p.ndim > 1:
        # TODO
        raise NotImplementedError

    raise NotImplementedError


def lasso_regression(q, X, y, tol=1e-8, reg=1):
    """
    Lasso Regression with Kelly penalty
    y : boolean array
    """
    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    a = np.zeros(X.shape[1])
    max_steps = 500
    for _ in range(max_steps):
        y_pred = (a * X).sum(1)
        grad = ((y / (q + y_pred) -
                (1 - y) / (1 - y_pred))[:, None] * X).sum(0)
        hess = ((y / (q + y_pred) ** 2 +
                (1 - y) / (1 - y_pred) ** 2)[:, None] * X * X).sum(0)
        step = (grad - reg) / hess
        a += step
        filt = a * hess < reg  # force to be positive with regularization
        a[filt] = 0
        if np.linalg.norm(step[~filt]) < tol:
            break
    else:
        raise ValueError
    # y_pred = (np.array([1, 0]) * X).sum(1)
    # C = (y * np.log(1 + y_pred / q) - (1 - y) * np.log(1 - y_pred)).sum()
    return a
