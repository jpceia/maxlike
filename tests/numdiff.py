"""
Finite-difference checks for the analytical derivatives of Funcs.

The structured tensors returned by Func.eval / grad / hess are expanded to
dense arrays (Tensor.toarray) and compared with central differences:
  - the gradient against finite differences of the value;
  - the Hessian against finite differences of the (analytical) gradient.
"""
import numpy as np


def dense(t):
    if hasattr(t, "toarray"):
        return np.asarray(t.toarray(), dtype=float)
    return np.asarray(t, dtype=float)


def _broadcast(a, b):
    shape = np.broadcast_shapes(a.shape, b.shape)
    return np.broadcast_to(a, shape), np.broadcast_to(b, shape)


def _perturb(params, i, idx, h):
    p = [np.array(q, dtype=float) for q in params]
    p[i][idx] += h
    return p


def fd_jacobian(fun, params, i, h=1e-6):
    """
    Central-difference derivative of fun(params) with respect to params[i].
    The result has shape params[i].shape + fun(params).shape.
    """
    p_i = np.asarray(params[i], dtype=float)
    out = None
    for idx in np.ndindex(p_i.shape):
        d = (dense(fun(_perturb(params, i, idx, h))) -
             dense(fun(_perturb(params, i, idx, -h)))) / (2 * h)
        if out is None:
            out = np.zeros(p_i.shape + d.shape)
        out[idx] = d
    return out


def derivative_errors(foo, params, use_eval=True, rtol=1e-5, atol=1e-7):
    """
    Returns a list of mismatches between the analytical derivatives of
    'foo' and finite differences (an empty list means everything agrees).

    use_eval=True checks foo.eval (the path used by MaxLike.fit);
    use_eval=False checks foo.grad and foo.hess.
    """
    params = [np.asarray(p, dtype=float) for p in params]
    n = len(params)

    if use_eval:
        _, grad, hess = foo.eval(params)

        def grad_at(q, j):
            return foo.eval(q)[1][j]
    else:
        grad = [foo.grad(params, i) for i in range(n)]
        hess = [[foo.hess(params, i, j) for j in range(i + 1)]
                for i in range(n)]

        def grad_at(q, j):
            return foo.grad(q, j)

    errors = []
    for i in range(n):
        A, F = _broadcast(dense(grad[i]), fd_jacobian(foo, params, i))
        if not np.allclose(A, F, rtol=rtol, atol=atol):
            errors.append(("grad", i, np.abs(A - F).max()))
        for j in range(i + 1):
            # hess[i][j]: x axes <-> params[i], y axes <-> params[j]
            A, F = _broadcast(
                dense(hess[i][j]),
                fd_jacobian(lambda q: grad_at(q, j), params, i))
            if not np.allclose(A, F, rtol=rtol, atol=atol):
                errors.append(("hess", i, j, np.abs(A - F).max()))
    return errors
