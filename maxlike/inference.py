"""
Inference and prediction on a fitted MaxLike problem: labelled parameters,
summary tables with confidence intervals, predictions with delta-method
standard errors, and simulation of new data from the fitted model.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def labels_from_values(values):
    """Axis labels carried by pandas inputs (None for plain arrays)."""
    if isinstance(values, pd.Series):
        return [values.index]
    if isinstance(values, pd.DataFrame):
        return [values.index, values.columns]
    return None


def axis_labels(mle, k):
    """Labels of each axis of parameter k (positions when not given)."""
    shape = mle.params[k].shape
    labels = mle.param_labels[k] if k < len(mle.param_labels) else None
    if labels is None:
        labels = [None] * len(shape)
    return [pd.Index(lab) if lab is not None else pd.RangeIndex(size)
            for lab, size in zip(labels, shape)]


def param_name(mle, k):
    names = getattr(mle, "param_names", [])
    if k < len(names) and names[k] is not None:
        return names[k]
    return "param %d" % k


def labeled(mle, k, values):
    """values (shaped like parameter k) as a labelled pandas object."""
    # fixed entries keep their (fixed) value
    values = np.asarray(np.ma.getdata(values), dtype=float)
    axes = axis_labels(mle, k)
    name = param_name(mle, k)
    if values.ndim == 0:
        return float(values)
    if values.ndim == 1:
        return pd.Series(values, index=axes[0], name=name)
    if values.ndim == 2:
        return pd.DataFrame(values, index=axes[0], columns=axes[1])
    index = pd.MultiIndex.from_product(axes)
    return pd.Series(values.ravel(), index=index, name=name)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def summary(mle, level=0.95):
    """
    One row per parameter entry: estimate, standard error, z statistic,
    two-sided p-value, confidence interval and whether it is fixed.
    """
    estimates = mle.get_params()
    errors = mle.std_error()
    q = norm.ppf(.5 + level / 2)
    lo_name = "ci_%g%%_lower" % (100 * level)
    hi_name = "ci_%g%%_upper" % (100 * level)
    frames = []
    for k, (est, err) in enumerate(zip(estimates, errors)):
        fixed = np.ma.getmaskarray(est)
        est = np.ma.getdata(est).astype(float)
        err = np.ma.filled(np.ma.asarray(err, dtype=float), np.nan)
        err = np.where(fixed, np.nan, err)
        axes = axis_labels(mle, k)
        if est.ndim == 0:
            labels = [""]
        elif est.ndim == 1:
            labels = list(axes[0])
        else:
            labels = list(pd.MultiIndex.from_product(axes))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = est / err
        frames.append(pd.DataFrame({
            "param": param_name(mle, k),
            "label": labels,
            "estimate": est.ravel(),
            "std_error": err.ravel(),
            "z": z.ravel(),
            "p_value": 2 * norm.sf(np.abs(z.ravel())),
            lo_name: (est - q * err).ravel(),
            hi_name: (est + q * err).ravel(),
            "fixed": fixed.ravel(),
        }))
    return pd.concat(frames, ignore_index=True).set_index(["param", "label"])


# --------------------------------------------------------------------------
# prediction with delta-method standard errors
# --------------------------------------------------------------------------
# above this many entries the dense Jacobian of the prediction would be too
# large: ask for a smaller target instead
_MAX_JACOBIAN_SIZE = 5 * 10 ** 7


def _dense(t):
    return np.asarray(t.toarray() if hasattr(t, "toarray") else t,
                      dtype=float)


def free_covariance(mle):
    """
    Covariance matrix of the free parameter entries (flattened, in the
    order used by the optimiser), from the Hessian of the last fit.
    Constraints are accounted for by the bordered (KKT) Hessian.
    """
    if not hasattr(mle, "flat_hess_"):
        raise RuntimeError("the model has not been fitted")
    hess = mle.flat_hess_
    hess = hess.toarray() if hasattr(hess, "toarray") else np.asarray(hess)
    n_free = sum(int(p.count()) for p in mle.params)
    return -np.linalg.inv(hess)[:n_free, :n_free]


def predict(mle, model=None, params=None, se=False):
    """
    Value of 'model' (default: the fitted model) at the fitted parameters,
    as a dense array. With se=True also returns delta-method standard
    errors, var(f) = J cov J^T with J the Jacobian of f with respect to
    the free parameters.
    """
    model = mle.model if model is None else model
    params = mle.params if params is None else params
    if not se:
        return _dense(model(params))

    value, grad, _ = model.eval(params)
    value = _dense(value)
    # check the size of the dense Jacobian before building it
    n_free = sum(int(p.count()) for p in params)
    n_values = max(int(value.size), 1)
    for d in grad:
        if hasattr(d, "_full_shape"):        # observation-indexed derivative
            n_values = max(n_values, int(np.prod(d._full_shape())))
        elif hasattr(d, "values"):
            p_dim = d.x_dim + d.y_dim
            n_values = max(n_values, int(np.prod(d.values.shape[p_dim:])))
    if n_free * n_values > _MAX_JACOBIAN_SIZE:
        raise ValueError(
            "the prediction has %d values and the model %d free parameters: "
            "the Jacobian is too large; pass a model that computes only the "
            "quantities of interest" % (n_values, n_free))
    blocks = [_dense(d) for d in grad]
    shape = np.broadcast_shapes(value.shape, *[
        b.shape[np.ndim(p):] for b, p in zip(blocks, params)])
    value = np.broadcast_to(value, shape)

    rows = []
    for p, b in zip(params, blocks):
        b = np.broadcast_to(b, p.shape + shape)
        rows.append(b[~p.mask].reshape(int(p.count()), -1))
    jac = np.concatenate(rows) if rows else np.zeros((0, value.size))
    cov = free_covariance(mle)
    var = np.einsum("ic,ij,jc->c", jac, cov, jac)
    return value.copy(), np.sqrt(np.maximum(var, 0)).reshape(shape)


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
def _data(mle, name):
    value = getattr(mle, name, None)
    if value is None:
        raise RuntimeError("the model has no data '%s': fit it first" % name)
    return np.asarray(value.values if hasattr(value, "values") else value)


def _counts(n):
    rounded = np.rint(n)
    if not np.allclose(n, rounded):
        raise ValueError("simulation needs integer counts N")
    return rounded.astype(np.int64)


def simulate(mle, rng=None, params=None):
    """
    Draws a new data set from the fitted model, with the same counts N as
    the data it was fitted on. Returns the keyword arguments of fit, so a
    parametric bootstrap is simply other.fit(**mle.simulate(rng)).
    """
    from . import maxlike as ml
    rng = np.random.default_rng(rng)
    params = mle.params if params is None else params
    y = _dense(mle.model(params))
    N = _data(mle, "N")
    y = np.broadcast_to(y, np.broadcast_shapes(y.shape, N.shape))
    n = _counts(N)

    if isinstance(mle, ml.Poisson):
        return {"N": N, "X": rng.poisson(n * np.exp(y)).astype(float)}

    if isinstance(mle, ml.Logistic):
        p = 1 / (1 + np.exp(-y))
        return {"N": N, "X": rng.binomial(n, p).astype(float)}

    if isinstance(mle, ml.NegativeBinomial):
        # the sum of n NB(r, mean m) draws is NB(n r, mean n m)
        r, m = mle.scale, np.exp(y)
        x = np.zeros(y.shape)
        pos = n > 0
        x[pos] = rng.negative_binomial(
            (n * r)[pos], np.broadcast_to(r / (r + m), y.shape)[pos])
        return {"N": N, "X": x}

    if type(mle) is ml.Finite:
        # outcomes along the distribution axes, totals kept per row
        dim = mle.dim
        lead = y.shape[:y.ndim - dim]
        prob = y.reshape(lead + (-1,))
        prob = prob / prob.sum(-1, keepdims=True)
        totals = n.reshape(lead + (-1,)).sum(-1)
        flat_p = prob.reshape(-1, prob.shape[-1])
        draws = np.array([rng.multinomial(t, q) for t, q in
                          zip(totals.ravel(), flat_p)])
        return {"N": draws.reshape(y.shape).astype(float)}

    raise NotImplementedError(
        "simulation is not implemented for %s" % type(mle).__name__)
