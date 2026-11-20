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


class Quadratic(Func):  # TODO : expand this class to allow more generic stuff

    def __init__(self, u=0, s=1):
        self.u = u
        self.s = s

    def __call__(self, params):
        val = 0
        for p in params:
            z = (np.asarray(p) - self.u) / self.s
            val += (z * z).sum()
        return Tensor(val)

    def grad(self, params, i):
        z = (np.asarray(params[i]) - self.u) / self.s
        return grad_tensor(2 * z / self.s, params, i, True).sum()

    def hess(self, params, i, j):
        if i != j:
            return Tensor(0)
        return hess_tensor(
            2 * np.ones_like(params[i]) / self.s / self.s,
            params, i, j, True, True).sum()

    def hess_sparse(self, params, i, j):
        """
        The same as hess, as a scipy.sparse matrix over the flattened
        parameters (None if null). Used when the Hessian of the problem is
        assembled as a sparse matrix, where the dense diagonal block
        returned by hess would not fit.
        """
        if i != j:
            return None
        size = np.asarray(params[i]).size
        return sp.diags(np.broadcast_to(2. / self.s / self.s, (size, )))


class X(Func):

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

    def __init__(self, vector):
        self.vector = np.asarray(vector)

    def __call__(self, params):
        return Tensor(self.vector)

    grad = null_func
    hess = null_func


class Scalar(Func):

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

    def __call__(self, params):
        return Tensor(np.log(params[0]))

    def grad(self, params, i):
        f = Tensor(1 / params[0])
        return grad_tensor(f, params, i, True, dim=f.dim)

    def hess(self, params, i, j):
        f = Tensor(-1 / np.square(params[0]))
        return hess_tensor(f, params, i, j, True, True, dim=f.dim)
