import numpy as np
import scipy.sparse as sp
from ..tensor import Tensor, IndexedTensor
from .func_base import Func, grad_tensor, hess_tensor, null_func


class Linear(Func):
    """
    Linear(w)(p_0, ..., p_k) = sum_i w_i * sum(p_i)

    A scalar function (no feature axes): it can be used as a constraint,
    a regularization or composed with other functions.
    """

    def __init__(self, weight_list=None):
        if isinstance(weight_list, (int, float)):
            weight_list = [weight_list]
        else:
            assert isinstance(weight_list, (list, tuple))
        self.weights = weight_list

    def __call__(self, params):
        return Tensor(sum((w * np.asarray(p)).sum()
                          for w, p in zip(self.weights, params)))

    def grad(self, params, i):
        # dense: every entry of params[i] contributes to the scalar output
        p = np.asarray(params[i])
        return Tensor(np.full(p.shape, self.weights[i], dtype=float),
                      x_dim=p.ndim)

    hess = null_func


class QuadraticForm(Func):
    """
    QuadraticForm(Q, u)(p_0, ..., p_k) = 1/2 * z' Q z,
        z = concatenation of the flattened p_i - u_i

    A scalar function (no feature axes), mostly used as a regularization:
    a Gaussian prior with precision matrix Q (minus its log density, up to a
    constant). Q can couple entries of one parameter (e.g. neighbours in
    time or space) and different parameters (e.g. the attack and the
    defence of the same team).

    Parameters
    ----------
    Q : array or scipy.sparse matrix, symmetric
        Precision matrix over the concatenation of the flattened
        parameters, in the order of the param_map they are given with.
    u : float, array, or list with one of them per parameter
        Centre, broadcast against each parameter.

    The Hessian is Q. It is kept sparse, and when Q is diagonal it is
    passed to the solver as a diagonal, as cheap as an L2 penalty.
    Subclasses build Q from the shapes of the parameters instead (see
    Quadratic and RandomWalk), by overriding _build(shapes); it is built
    once per set of shapes.
    """

    def __init__(self, Q=None, u=0.):
        self._Q = None if Q is None else sp.csr_matrix(Q)
        self.u = u
        self._cache = {}

    def _build(self, shapes):
        """Precision matrix for parameters of the given shapes."""
        size = sum(int(np.prod(sh)) for sh in shapes)
        if self._Q is None or self._Q.shape != (size, size):
            raise ValueError(
                "QuadraticForm: Q must be %d x %d for parameters of shapes "
                "%s" % (size, size, shapes))
        return self._Q

    def _form(self, params):
        # (Q as CSR, diagonal of Q or None, offset of each parameter)
        shapes = tuple(np.shape(p) for p in params)
        if shapes not in self._cache:
            Q = sp.csr_matrix(self._build(shapes))
            diag = Q.diagonal()
            is_diag = Q.nnz == np.count_nonzero(diag) and \
                (sp.triu(Q, 1).nnz + sp.tril(Q, -1).nnz) == 0
            sizes = [int(np.prod(sh)) for sh in shapes]
            self._cache[shapes] = (Q, diag if is_diag else None,
                                   np.concatenate([[0], np.cumsum(sizes)]))
        return self._cache[shapes]

    def _z(self, params):
        u = self.u if isinstance(self.u, (list, tuple)) else \
            [self.u] * len(params)
        return np.concatenate([
            (np.asarray(p, dtype=float) - np.asarray(c, dtype=float)).ravel()
            for p, c in zip(params, u)])

    def __call__(self, params):
        Q, diag, _ = self._form(params)
        z = self._z(params)
        Qz = diag * z if diag is not None else Q @ z
        return Tensor(.5 * z @ Qz)

    def grad(self, params, i):
        Q, diag, off = self._form(params)
        z = self._z(params)
        sl = slice(off[i], off[i + 1])
        g = diag[sl] * z[sl] if diag is not None else Q[sl] @ z
        return grad_tensor(g.reshape(np.shape(params[i])), params, i)

    def hess(self, params, i, j):
        Q, diag, off = self._form(params)
        shape_i, shape_j = np.shape(params[i]), np.shape(params[j])
        if diag is not None:
            if i != j:
                return Tensor(0)
            # diagonal: stored as such, summed over the (mapped) axes
            return hess_tensor(diag[off[i]:off[i + 1]].reshape(shape_i),
                               params, i, j, True, True).sum()
        block = Q[off[i]:off[i + 1], off[j]:off[j + 1]]
        if block.nnz == 0:
            return Tensor(0)
        return hess_tensor(block.toarray().reshape(shape_i + shape_j),
                           params, i, j)

    def hess_sparse(self, params, i, j):
        """
        The same as hess, as a scipy.sparse matrix over the flattened
        parameters (None if null). Used when the Hessian of the problem is
        assembled as a sparse matrix, where a dense block would not fit.
        """
        Q, _, off = self._form(params)
        block = Q[off[i]:off[i + 1], off[j]:off[j + 1]]
        return block if block.nnz else None


class Quadratic(QuadraticForm):
    """
    Quadratic(u, s)(p_0, ..., p_k) = sum_i sum(((p_i - u) / s) ** 2)

    An L2 penalty: pulls every entry of the parameters towards u, with a
    scale s (a Gaussian prior with mean u and standard deviation
    s / sqrt(2), up to a constant). A QuadraticForm with the diagonal
    precision Q = 2 / s^2.

    Parameters
    ----------
    u : float or array
        Centre, broadcast against the parameters.
    s : float or array
        Scale, broadcast against the parameters.
    """

    def __init__(self, u=0, s=1):
        super().__init__(u=u)
        self.s = s

    def _build(self, shapes):
        return sp.diags(np.concatenate([
            np.broadcast_to(2. / np.square(self.s), sh).ravel()
            for sh in shapes]))


class RandomWalk(QuadraticForm):
    """
    RandomWalk(s, axis)(p_0, ..., p_k) =
        1/2 * sum_i sum_t ((p_i[..., t] - p_i[..., t - 1]) / s[t]) ** 2

    Penalty of a Gaussian random walk along `axis` (time): minus the log
    density of the increments, up to a constant. Used as a regularization,
    it lets a parameter with a time axis (e.g. the attack of each team in
    each period) move from one period to the next, with steps of typical
    size s. A small s tends to a constant parameter (a static model), a
    large s to independent fits per period.

    A QuadraticForm with Q = D' W D, D the differences along `axis` and W
    the diagonal of 1 / s^2: a sparse second-difference matrix.

    Parameters
    ----------
    s : float or array
        Standard deviation of each step, broadcast against the differences
        (shape of p with `axis` one shorter). With unequal gaps between
        periods, use s = sigma * sqrt(gap).
    axis : int
        Time axis of the parameters.
    """

    def __init__(self, s=1., axis=-1):
        super().__init__()
        self.s = s
        self.axis = axis

    def _block(self, shape):
        # D' W D over one flattened parameter
        size = int(np.prod(shape))
        pos = np.moveaxis(np.arange(size).reshape(shape), self.axis, -1)
        steps = list(shape)
        steps[self.axis] -= 1
        w = np.moveaxis(np.broadcast_to(
            np.asarray(self.s, dtype=float) ** -2., tuple(steps)),
            self.axis, -1)
        rows = np.arange(pos[..., 1:].size)
        d = sp.csr_matrix(
            (np.concatenate([np.ones(rows.size), -np.ones(rows.size)]),
             (np.concatenate([rows, rows]),
              np.concatenate([pos[..., 1:].ravel(), pos[..., :-1].ravel()]))),
            shape=(rows.size, size))
        return d.T @ sp.diags(w.ravel()) @ d

    def _build(self, shapes):
        return sp.block_diag([self._block(sh) for sh in shapes])


class X(Func):
    """
    X()(p) = p

    The identity: the output has the shape of the parameter, one feature
    axis per axis of p. The derivative is diagonal and only its diagonal is
    stored. Combined with Sum.add, it maps a parameter onto feature axes of
    a grid model (e.g. attack[i] onto the axis of the home team).
    """

    def __call__(self, params):
        return Tensor(params[0])

    def grad(self, params, i):
        return grad_tensor(np.ones_like(params[0]), params, i, True)

    hess = null_func


class Take(Func):
    """
    Take(index)(p) = p[index]            (times `weight`, if given)

    Gathers one parameter entry per observation: index[o] is the entry of p
    that observation o depends on (a team, a player, a student, ...). It is
    the observation-indexed counterpart of X(): the data are listed one
    observation per row instead of on the grid of all the combinations of
    the features, so the size of the problem follows the number of
    observations.

    Parameters
    ----------
    index : int array, or tuple of int arrays
        Entry of the parameter for each observation. For a parameter with
        several axes, a tuple with one array per axis (as in p[i, j]).
    weight : array (optional)
        Per-observation factor (a covariate), broadcast against index.

    The derivative is an incidence matrix with one non-zero per observation,
    stored as an IndexedTensor.
    """

    def __init__(self, index, weight=None):
        if isinstance(index, tuple):
            index = tuple(np.broadcast_arrays(*index))
        else:
            index = (np.asarray(index), )
        for arr in index:
            if arr.dtype.kind not in "iu":
                raise TypeError("Take: the index must be an integer array")
        self.index = index
        self.weight = None if weight is None else \
            np.broadcast_to(np.asarray(weight, dtype=float), index[0].shape)
        self._flat = (None, None)    # (parameter shape, flat index)

    def _flat_index(self, shape):
        # position of every observation in the flattened parameter
        if self._flat[0] == shape:
            return self._flat[1]
        if len(shape) != len(self.index):
            raise ValueError(
                "Take: %d index array(s) for a parameter with %d axes" %
                (len(self.index), len(shape)))
        coords = []
        for arr, size in zip(self.index, shape):
            arr = np.where(arr < 0, arr + size, arr)
            if arr.size and (arr.min() < 0 or arr.max() >= size):
                raise IndexError(
                    "Take: index out of bounds for an axis of size %d" % size)
            coords.append(arr)
        flat = np.asarray(np.ravel_multi_index(coords, shape), dtype=np.intp)
        flat.flags["WRITEABLE"] = False
        self._flat = (shape, flat)
        return flat

    def __call__(self, params):
        p = np.asarray(params[0])
        val = p.reshape(-1)[self._flat_index(p.shape)]
        if self.weight is not None:
            val = val * self.weight
        return Tensor(val)

    def grad(self, params, i):
        shape = np.shape(np.asarray(params[i]))
        idx = self._flat_index(shape)
        vals = self.weight
        if vals is None:
            vals = np.broadcast_to(1.0, idx.shape)
        return IndexedTensor([(idx, None, vals)], x_shape=shape, n=idx.ndim)

    hess = null_func

    def graph_label(self):
        # label of this function in a model graph (see maxlike.graph)
        size = self._flat[0]
        if size is None:
            size = tuple(int(arr.max()) + 1 if arr.size else 0
                         for arr in self.index)
        return "Take(n_obs=%s, size=%s%s)" % (
            "x".join(map(str, self.index[0].shape)),
            "x".join(map(str, size)),
            "" if self.weight is None else ", weighted")


class Constant(Func):
    """
    Constant(v)(params) = v

    A fixed array that does not depend on the parameters (zero derivatives):
    an offset of a linear predictor, known log-exposures, and so on.

    Parameters
    ----------
    vector : array-like
        The value returned.
    """

    def __init__(self, vector):
        self.vector = np.asarray(vector)

    def __call__(self, params):
        return Tensor(self.vector)

    grad = null_func
    hess = null_func


class Scalar(Func):
    """
    Scalar()(p) = p

    The identity for a scalar parameter (a 0-d parameter, e.g. the home
    advantage or a global intercept): the output has no feature axes, so
    Sum broadcasts it over all of them. For a parameter with one value per
    observation or per cell, use X or Vector.
    """

    def __call__(self, params):
        return Tensor(params[0])

    def grad(self, params, i):
        return Tensor(1)

    hess = null_func


class Vector(Func):
    """
    Vector(v)(p) = p * v

    p is either a scalar (the output is v scaled by p) or an array with the
    shape of the output (element-wise product, v broadcast against p).
    """

    def __init__(self, vector):
        self.vector = np.asarray(vector)

    def __call__(self, params):
        return Tensor(np.asarray(params[0]) * self.vector)

    def grad(self, params, i):
        p = np.asarray(params[i])
        if p.ndim == 0:
            return grad_tensor(self.vector, params, i, None)
        shape = np.broadcast_shapes(p.shape, self.vector.shape)
        if p.shape != shape:
            raise ValueError(
                "Vector: the parameter must be a scalar or have the shape "
                "of the output %s, got %s" % (shape, p.shape))
        # element-wise product: the derivative is diagonal, equal to v
        return grad_tensor(np.broadcast_to(self.vector, shape),
                           params, i, True)

    hess = null_func


class Exp(Func):
    """
    Exp()(p) = exp(p), element-wise.

    Usually composed with a linear predictor, Exp() @ model, to turn a log
    rate into a rate. The derivatives are diagonal.
    """

    def __call__(self, params):
        return Tensor(np.exp(params[0]))

    def grad(self, params, i):
        f = Tensor(np.exp(params[0]))
        return grad_tensor(f, params, i, True, dim=f.dim)

    def hess(self, params, i, j):
        f = Tensor(np.exp(params[0]))
        return hess_tensor(f, params, i, j, True, True, dim=f.dim)

    def eval(self, params):
        val = Tensor(np.exp(params[0]))
        grad = grad_tensor(val, params, 0, True, dim=val.dim)
        hess = hess_tensor(val, params, 0, 0, True, True, dim=val.dim)
        return val, [grad], [[hess]]


class Log(Func):
    """
    Log()(p) = log(p), element-wise (p > 0).

    The derivatives are diagonal.
    """

    def __call__(self, params):
        return Tensor(np.log(params[0]))

    def grad(self, params, i):
        f = Tensor(1 / params[0])
        return grad_tensor(f, params, i, True, dim=f.dim)

    def hess(self, params, i, j):
        f = Tensor(-1 / np.square(params[0]))
        return hess_tensor(f, params, i, j, True, True, dim=f.dim)
