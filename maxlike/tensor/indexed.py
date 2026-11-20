"""
Observation-indexed derivatives.

A model whose data are listed one observation per row gathers parameter
entries per observation, y[o] = p[index[o]] (see maxlike.func.Take). Its
derivative with respect to p is an incidence matrix with one non-zero per
observation. Tensor can only store a derivative that is dense or diagonal in
a parameter axis (x_map / y_map), so that structure is stored here instead:

    IndexedTensor   a sum of terms (x_idx, y_idx, vals), meaning

                        T[a, b, w] = sum_terms vals[w] * [a == x_idx[w]]
                                                       * [b == y_idx[w]]

                    where w runs over the feature (observation) and
                    distribution axes, and a, b are flat positions in
                    parameters i and j. A gradient has no y_idx; one of the
                    two sides is also absent for a scalar parameter.

    SparseBlock     the sum of a Hessian term over the observations: a
                    scipy.sparse (param_i x param_j) matrix.

Both plug into the arithmetic of Tensor / GenericTensor (which dispatch to
them when they meet an operand they do not know), so the likelihoods and
Compose / Sum / Product work unchanged.
"""
import numpy as np
import scipy.sparse as sp
from functools import lru_cache
from .tensor import (
    BaseTensor, Tensor, GenericTensor, TensorOp, InvalidOperation,
    _is_zero, _contract)


_ADDITIVE = (TensorOp.ADD, TensorOp.SUB, TensorOp.RSUB)

# `t op self` written as `self op' t`
_REVERSED = {
    TensorOp.ADD: TensorOp.ADD,
    TensorOp.MUL: TensorOp.MUL,
    TensorOp.SUB: TensorOp.RSUB,
    TensorOp.RSUB: TensorOp.SUB,
    TensorOp.DIV: TensorOp.RDIV,
    TensorOp.RDIV: TensorOp.DIV,
}

_UFUNC_OPS = {
    'add': TensorOp.ADD,
    'subtract': TensorOp.RSUB,
    'multiply': TensorOp.MUL,
    'true_divide': TensorOp.RDIV,
    'divide': TensorOp.RDIV,
}


@lru_cache(maxsize=8)
def _arange(size):
    arr = np.arange(size, dtype=np.intp)
    arr.flags["WRITEABLE"] = False
    return arr


def _identity_idx(shape, ndim):
    """
    Flat position of every entry of an array with the given shape, laid
    along the first axes of an array with ndim axes. It is a view of a
    cached array, which _is_identity recognises.
    """
    size = int(np.prod(shape, dtype=np.intp))
    return _arange(size).reshape(tuple(shape) + (1,) * (ndim - len(shape)))


def _is_identity(idx, size):
    return idx.size == size and idx.base is _arange(size) and \
        idx.flags["C_CONTIGUOUS"]


def _key(idx):
    # two index arrays with the same buffer, shape and strides are equal
    if idx is None:
        return None
    return (idx.__array_interface__['data'][0], idx.shape, idx.strides)


def _pad(arr, count):
    """Appends `count` broadcast axes"""
    if arr is None or count == 0:
        return arr
    return arr.reshape(arr.shape + (1,) * count)


def _move_axes(arr, start, count, where, newsize):
    """
    The `count` axes of arr starting at `start` are sent to the positions
    `where` of a group of `newsize` axes; the others are broadcast axes.
    """
    if arr is None:
        return None
    perm = list(range(arr.ndim))
    perm[start:start + count] = [start + k for k in np.argsort(where)]
    idx = [slice(None)] * start
    idx += [slice(None) if k in where else None for k in range(newsize)]
    idx += [slice(None)] * (arr.ndim - start - count)
    return arr.transpose(perm)[tuple(idx)]


def _size(shape):
    return int(np.prod(shape, dtype=np.intp)) if shape is not None else 1


def _as_scalar(other):
    """The value of a scalar operand, or None if `other` is not a scalar"""
    if isinstance(other, Tensor):
        other = other.values
    if np.isscalar(other):
        return other
    if isinstance(other, np.ndarray) and other.ndim == 0:
        return other[()]
    return None


class _Dispatched(BaseTensor):
    """Operators shared by the tensors of this module"""

    __slots__ = []

    def _rbin_op(self, other, op_type):
        # called by Tensor / GenericTensor for `other op self`
        return self.bin_op(other, _REVERSED[op_type])

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        try:
            op_type = _UFUNC_OPS[ufunc.__name__]
        except KeyError:
            raise InvalidOperation('Invalid ufunc:', ufunc.__name__)
        return self.bin_op(inputs[0], op_type)


class IndexedTensor(_Dispatched):
    """
    Derivative that is sparse in the parameter axes: every position of the
    feature / distribution axes depends on a few parameter entries only.

    terms : tuple of (x_idx, y_idx, vals)
        vals has n + dim axes (features, then distribution axes); x_idx and
        y_idx are arrays of flat positions in the parameters, with the same
        number of axes, or None when that side is absent. All broadcast
        against each other.
    x_shape, y_shape : shape of parameters i and j (None if absent).
    """

    __slots__ = ['terms', 'x_shape', 'y_shape']

    def __init__(self, terms, x_shape=None, y_shape=None, n=1, dim=0):
        x_shape = None if x_shape is None else tuple(x_shape)
        y_shape = None if y_shape is None else tuple(y_shape)
        super(IndexedTensor, self).__init__(
            len(x_shape or ()), len(y_shape or ()), n, dim)
        object.__setattr__(self, 'terms', tuple(terms))
        object.__setattr__(self, 'x_shape', x_shape)
        object.__setattr__(self, 'y_shape', y_shape)

    def _new(self, terms, n=None, dim=None):
        return IndexedTensor(terms, self.x_shape, self.y_shape,
                             self.n if n is None else n,
                             self.dim if dim is None else dim)

    def _map_vals(self, foo):
        return self._new([(xi, yi, foo(v)) for xi, yi, v in self.terms])

    def _map_arrays(self, foo, n=None, dim=None):
        return self._new([(foo(xi), foo(yi), foo(v))
                          for xi, yi, v in self.terms], n, dim)

    def _full_shape(self):
        """Shape of the feature and distribution axes"""
        return np.broadcast_shapes(*(
            arr.shape for term in self.terms for arr in term
            if arr is not None))

    # ----------------------------------------------------------------------
    # conversions
    # ----------------------------------------------------------------------
    @staticmethod
    def from_tensor(t):
        """
        Indexed form of a Tensor. A parameter axis that is diagonal along a
        feature axis becomes an index; a dense parameter axis becomes one
        term per entry.
        """
        p = t.x_dim + t.y_dim
        w_shape = t.values.shape[p:]
        w_dim = len(w_shape)

        def side(offset, count, mapping):
            # shape of the parameter, index of the diagonal axes and the
            # (axis, stride) of the dense ones
            if count == 0:
                return None, None, [], []
            mapping = list(mapping) + [-1] * (count - len(mapping))
            shape = tuple(w_shape[f] if f >= 0 else t.values.shape[offset + k]
                          for k, f in enumerate(mapping))
            strides = [_size(shape[k + 1:]) for k in range(count)]
            if mapping == list(range(count)):
                return shape, _identity_idx(shape, w_dim), [], []
            base = np.zeros((1,) * w_dim, dtype=np.intp)
            dense, dense_strides = [], []
            for k, f in enumerate(mapping):
                if f < 0:
                    dense.append(k)
                    dense_strides.append(strides[k])
                else:
                    coord_shape = [1] * w_dim
                    coord_shape[f] = w_shape[f]
                    base = base + strides[k] * \
                        _arange(w_shape[f]).reshape(coord_shape)
            return shape, base, dense, dense_strides

        x_shape, x_base, x_dense, x_strides = side(0, t.x_dim, t.x_map)
        y_shape, y_base, y_dense, y_strides = side(t.x_dim, t.y_dim, t.y_map)

        terms = []
        for cx in np.ndindex(*(x_shape[k] for k in x_dense)):
            for cy in np.ndindex(*(y_shape[k] for k in y_dense)):
                sel = [0] * p
                x_idx, y_idx = x_base, y_base
                if cx:
                    for k, c in zip(x_dense, cx):
                        sel[k] = c
                    x_idx = x_base + int(np.dot(cx, x_strides))
                if cy:
                    for k, c in zip(y_dense, cy):
                        sel[t.x_dim + k] = c
                    y_idx = y_base + int(np.dot(cy, y_strides))
                terms.append((x_idx, y_idx, t.values[tuple(sel)]))
        return IndexedTensor(terms, x_shape, y_shape, t.n, t.dim)

    def toarray(self):
        shape = self._full_shape()
        size = _size(shape)
        out = np.zeros((_size(self.x_shape), _size(self.y_shape), size))
        pos = _arange(size)
        for x_idx, y_idx, vals in self.terms:
            np.add.at(out, (
                0 if x_idx is None else np.broadcast_to(x_idx, shape).ravel(),
                0 if y_idx is None else np.broadcast_to(y_idx, shape).ravel(),
                pos), np.broadcast_to(vals, shape).ravel())
        return out.reshape(
            (self.x_shape or ()) + (self.y_shape or ()) + shape)

    def to_tensor(self):
        """Dense Tensor with the same values (memory: parameters x cells)"""
        return Tensor(self.toarray(), x_dim=self.x_dim, y_dim=self.y_dim,
                      dim=self.dim)

    # ----------------------------------------------------------------------
    # sums
    # ----------------------------------------------------------------------
    def _scatter(self, shape, axes):
        """
        Flattened (x index, y index, position, values) of every term, ready
        to be accumulated over `axes` of the feature / distribution axes
        (which have the given broadcast shape). `position` is the flat
        position along the axes that are kept (None if there are none).
        """
        ndim = len(shape)
        keep = [k for k in range(ndim) if k not in axes]
        pos = None
        if keep:
            pos = _arange(_size([shape[k] for k in keep])).reshape(
                [shape[k] if k in keep else 1 for k in range(ndim)])
        for x_idx, y_idx, vals in self.terms:
            # the axes no index depends on are summed right away
            free = tuple(k for k in axes
                         if (x_idx is None or x_idx.shape[k] == 1) and
                            (y_idx is None or y_idx.shape[k] == 1))
            if free:
                # values that broadcast along an axis count once per entry
                mult = _size([shape[k] for k in free if vals.shape[k] == 1])
                reduce = tuple(k for k in free if vals.shape[k] != 1)
                if reduce:
                    vals = vals.sum(reduce, keepdims=True)
                if mult != 1:
                    vals = vals * mult
            rest = tuple(1 if k in free else shape[k] for k in range(ndim))
            yield tuple(
                None if arr is None else np.broadcast_to(arr, rest).ravel()
                for arr in (x_idx, y_idx, pos, vals))

    def sum(self, sum_val=True, sum_dim=True):
        """
        Sum over the feature and / or distribution axes.

        Over everything, a gradient becomes a dense Tensor shaped like the
        parameter (np.bincount) and a Hessian term a SparseBlock.
        """
        if isinstance(sum_val, int) and not isinstance(sum_val, bool):
            sum_val = [sum_val]
        if isinstance(sum_val, (list, tuple)):
            if len(set(sum_val)) == self.n:
                sum_val = True
            elif len(sum_val) == 0:
                sum_val = False
            else:
                raise NotImplementedError(
                    "IndexedTensor: sum over part of the feature axes")
        sum_dim = sum_dim is True and self.dim > 0

        if sum_val is False and not sum_dim:
            return self

        shape = self._full_shape()
        n = self.n

        if sum_val is False:
            # distribution axes only: the structure is kept
            axes = tuple(range(n, n + self.dim))
            e_size = _size(shape[n:])
            terms = []
            for x_idx, y_idx, vals in self.terms:
                for idx in (x_idx, y_idx):
                    if idx is not None and _size(idx.shape[n:]) != 1:
                        raise NotImplementedError(
                            "IndexedTensor: sum over distribution axes "
                            "the index depends on")
                mult = e_size / _size(vals.shape[n:])
                vals = vals.sum(axes)
                if mult != 1:
                    vals = vals * mult
                terms.append((
                    None if x_idx is None else x_idx.reshape(x_idx.shape[:n]),
                    None if y_idx is None else y_idx.reshape(y_idx.shape[:n]),
                    vals))
            return self._new(terms, dim=0)

        if not sum_dim and self.dim > 0:
            # features only, keeping the distribution axes: dense result
            e_shape = shape[n:]
            out = np.zeros((_size(self.x_shape), _size(self.y_shape),
                            _size(e_shape)))
            for x_idx, y_idx, pos, vals in self._scatter(
                    shape, tuple(range(n))):
                np.add.at(out, (0 if x_idx is None else x_idx,
                                0 if y_idx is None else y_idx, pos), vals)
            return Tensor(
                out.reshape((self.x_shape or ()) + (self.y_shape or ()) +
                            e_shape),
                x_dim=self.x_dim, y_dim=self.y_dim, dim=self.dim)

        # everything
        axes = tuple(range(len(shape)))
        x_size, y_size = _size(self.x_shape), _size(self.y_shape)
        if self.x_shape is None or self.y_shape is None:
            out = np.zeros(x_size * y_size)
            for x_idx, y_idx, _, vals in self._scatter(shape, axes):
                out += np.bincount(x_idx if y_idx is None else y_idx,
                                   weights=vals, minlength=out.size)
            if self.y_shape is None:
                return Tensor(out.reshape(self.x_shape), x_dim=self.x_dim)
            return Tensor(out.reshape(self.y_shape), y_dim=self.y_dim)

        rows, cols, data = [], [], []
        for (x_raw, y_raw, _), (x_idx, y_idx, _, vals) in zip(
                self.terms, self._scatter(shape, axes)):
            if x_size == y_size and _key(x_raw) == _key(y_raw):
                # same entry on both sides: a diagonal
                diag = np.bincount(x_idx, weights=vals, minlength=x_size)
                x_idx = y_idx = np.flatnonzero(diag)
                vals = diag[x_idx]
            rows.append(x_idx)
            cols.append(y_idx)
            data.append(vals)
        return SparseBlock(
            sp.coo_matrix((np.concatenate(data),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(x_size, y_size)),
            self.x_shape, self.y_shape)

    # ----------------------------------------------------------------------
    # structure
    # ----------------------------------------------------------------------
    def expand(self, where, newsize, dim=False):
        where = list(where or [])
        if dim is True:
            assert len(where) == self.dim
            if where == list(range(newsize)):
                return self
            return self._map_arrays(
                lambda a: _move_axes(a, self.n, self.dim, where, newsize),
                dim=newsize)
        assert len(where) == self.n
        if where == list(range(newsize)):
            return self
        return self._map_arrays(
            lambda a: _move_axes(a, 0, self.n, where, newsize), n=newsize)

    def flip(self, where, dim=False):
        if where is None or where == []:
            return self
        if isinstance(where, int):
            where = [where]
        offset = self.n if dim is True else 0

        def flip(arr):
            if arr is None:
                return None
            for k in where:
                if arr.shape[offset + k] > 1:
                    arr = np.flip(arr, offset + k)
            return arr
        return self._map_arrays(flip)

    def transpose(self):
        return IndexedTensor(
            [(yi, xi, v) for xi, yi, v in self.terms],
            self.y_shape, self.x_shape, self.n, self.dim)

    def drop_dim(self):
        if self.dim == 0:
            return self
        return self._new(self.terms, n=self.n + self.dim, dim=0)

    def __neg__(self):
        return self._map_vals(np.negative)

    # ----------------------------------------------------------------------
    # chain rule
    # ----------------------------------------------------------------------
    def _chain(self, other):
        """
        Contraction over the feature axes N of `other`, the x axes of self:

            result[X, Y, M, E] = sum_N other[X, Y_o, N] * self[N, Y_s, M, E]

        Every cell of self reads a few cells N (its x index), and each of
        those a few parameter entries (the indexes of other): the result
        gathers the indexes and values of other at the x index of self.
        """
        if _is_zero(self) or _is_zero(other):
            return Tensor(0)

        if isinstance(other, GenericTensor):
            return sum((self._chain(el) for el in other.elements), Tensor(0))

        if isinstance(other, Tensor):
            if other.x_dim == 0 and other.y_dim == 0:
                other = IndexedTensor(
                    [(None, None, other.values)], n=other.n, dim=other.dim)
            else:
                other = IndexedTensor.from_tensor(other)

        n_shape = self.x_shape
        if n_shape is None or other.dim != 0 or other.n != len(n_shape) or \
                (other.y_shape is not None and self.y_shape is not None):
            raise InvalidOperation(
                "IndexedTensor: incoherent chain-rule contraction")
        n_size = _size(n_shape)

        terms = []
        for ox, oy, ov in other.terms:
            # the arrays of other over the whole of N, flattened
            flat = [None if a is None else np.broadcast_to(a, n_shape)
                    for a in (ox, oy, ov)]
            for sx, sy, sv in self.terms:
                if _is_identity(sx, n_size):
                    gx, gy, gv = (None if a is None else a.reshape(sx.shape)
                                  for a in flat)
                else:
                    gx, gy, gv = (None if a is None else a.ravel()[sx]
                                  for a in flat)
                terms.append((gx, sy if gy is None else gy, gv * sv))

        x_shape = other.x_shape
        y_shape = self.y_shape if other.y_shape is None else other.y_shape
        if x_shape is None and y_shape is None:
            return Tensor(sum(v for _, _, v in terms), dim=self.dim)
        return IndexedTensor(terms, x_shape, y_shape,
                             self.n, self.dim)._merged()

    def _contract_into(self, f):
        """
        Called by Tensor.dot_left / dot_right: contraction of self (the
        derivative of an inner function) with the Tensor f (the derivative
        of an outer function).

        When f is diagonal in its input (an element-wise function: Exp,
        Logistic, a pmf with a distribution axis, ...) the index structure
        is kept. Otherwise self is expanded to a dense Tensor.
        """
        if f.x_dim and len(f.x_map) == f.x_dim and min(f.x_map) >= 0 and \
                (f.y_dim == 0 or
                 len(f.y_map) == f.y_dim and min(f.y_map) >= 0):
            g = IndexedTensor.from_tensor(f)
            if g.x_shape == self._full_shape():
                return g._chain(self)
        return _contract(self.to_tensor(), f)

    def dot_left(self, other):
        if (self.x_dim != other.n) & (self.y_dim == other.n):
            return self.transpose()._chain(other)
        return self._chain(other)

    def dot_right(self, other):
        return self._chain(other)

    def dot(self, other):
        return self.dot_left(other)

    # ----------------------------------------------------------------------
    # arithmetic
    # ----------------------------------------------------------------------
    def _merged(self):
        """Adds up the terms that have the same indexes"""
        if len(self.terms) < 2:
            return self
        merged = {}
        for x_idx, y_idx, vals in self.terms:
            key = (_key(x_idx), _key(y_idx))
            if key in merged:
                merged[key] = (x_idx, y_idx, merged[key][2] + vals)
            else:
                merged[key] = (x_idx, y_idx, vals)
        if len(merged) == len(self.terms):
            return self
        return self._new(merged.values())

    def _with_dim(self, dim):
        # pads a tensor without distribution axes with `dim` broadcast axes
        if self.dim == dim:
            return self
        assert self.dim == 0
        return self._map_arrays(lambda a: _pad(a, dim), dim=dim)

    def bin_op(self, other, op_type):

        if isinstance(other, GenericTensor):
            if op_type in _ADDITIVE:
                res = -self if op_type == TensorOp.RSUB else self
                el_op = TensorOp.SUB if op_type == TensorOp.SUB \
                    else TensorOp.ADD
                for el in other.elements:
                    res = res.bin_op(el, el_op)
                return res
            if op_type == TensorOp.MUL:
                return sum((self.bin_op(el, op_type)
                            for el in other.elements), Tensor(0))
            raise InvalidOperation

        if isinstance(other, np.ndarray) and other.ndim > 0:
            other = Tensor(other)

        scalar = _as_scalar(other)
        if scalar is not None:
            if scalar == 0:
                if op_type in (TensorOp.ADD, TensorOp.SUB):
                    return self
                if op_type == TensorOp.RSUB:
                    return -self
                if op_type in (TensorOp.MUL, TensorOp.RDIV):
                    return Tensor(0)
                raise ZeroDivisionError
            if op_type in (TensorOp.MUL, TensorOp.DIV):
                if scalar == 1:
                    return self
                return self._map_vals(lambda v: op_type.value(v, scalar))
            # a constant added to a sparse derivative would fill it
            raise InvalidOperation

        if isinstance(other, Tensor):
            if other.x_dim == 0 and other.y_dim == 0:
                return self._op_values(other, op_type)
            other = IndexedTensor.from_tensor(other)

        if isinstance(other, IndexedTensor):
            return self._op_indexed(other, op_type)

        if isinstance(other, SparseBlock):
            raise InvalidOperation

        raise ValueError

    def _op_values(self, other, op_type):
        # product with / division by values over the cells
        if op_type not in (TensorOp.MUL, TensorOp.DIV):
            raise InvalidOperation
        assert (other.n == 0) | (other.n == self.n)
        assert (self.dim == 0) | (other.dim == 0) | (self.dim == other.dim)
        dim = max(self.dim, other.dim)
        values = other.values[other._reshape_idx(0, 0, self.n, dim)]
        return self._with_dim(dim)._map_vals(
            lambda v: op_type.value(v, values))

    def _op_indexed(self, other, op_type):
        assert self.n == other.n
        assert (self.dim == 0) | (other.dim == 0) | (self.dim == other.dim)
        dim = max(self.dim, other.dim)
        a = self._with_dim(dim)
        b = other._with_dim(dim)

        if op_type in _ADDITIVE:
            if a.x_shape != b.x_shape or a.y_shape != b.y_shape:
                raise InvalidOperation(
                    "IndexedTensor: sum of derivatives with respect to "
                    "different parameters")
            if op_type == TensorOp.SUB:
                b = -b
            elif op_type == TensorOp.RSUB:
                a = -a
            return a._new(a.terms + b.terms)._merged()

        if op_type == TensorOp.MUL:
            # outer product of the derivatives with respect to parameter i
            # and to parameter j: pairs of entries, one per observation
            if (a.x_shape is not None and b.x_shape is not None) or \
                    (a.y_shape is not None and b.y_shape is not None):
                raise InvalidOperation(
                    "IndexedTensor: element-wise product of derivatives "
                    "with respect to the same parameter axes")
            terms = [(bx if ax is None else ax,
                      by if ay is None else ay, av * bv)
                     for ax, ay, av in a.terms for bx, by, bv in b.terms]
            return IndexedTensor(
                terms,
                b.x_shape if a.x_shape is None else a.x_shape,
                b.y_shape if a.y_shape is None else a.y_shape,
                a.n, dim)._merged()

        raise InvalidOperation

    def __str__(self):
        return "indexed, %d term(s), shape: [%s, %s, %d, %d]" % (
            len(self.terms), self.x_shape, self.y_shape, self.n, self.dim)


class SparseBlock(_Dispatched):
    """
    Block of a gradient-of-gradient (Hessian) summed over the observations:
    a sparse matrix with one row per entry of parameter i and one column per
    entry of parameter j.

    matrix : scipy.sparse matrix (duplicate entries are summed)
    """

    __slots__ = ['matrix', 'x_shape', 'y_shape']

    def __init__(self, matrix, x_shape, y_shape):
        super(SparseBlock, self).__init__(len(x_shape), len(y_shape))
        object.__setattr__(self, 'matrix', matrix)
        object.__setattr__(self, 'x_shape', tuple(x_shape))
        object.__setattr__(self, 'y_shape', tuple(y_shape))

    @property
    def values(self):
        # dense values, as in Tensor
        return self.toarray()

    def toarray(self):
        return np.asarray(self.matrix.toarray()).reshape(
            self.x_shape + self.y_shape)

    def sum(self, sum_val=True, sum_dim=True):
        return self

    def expand(self, where, newsize, dim=False):
        if newsize == 0:
            return self
        raise NotImplementedError

    def flip(self, where, dim=False):
        if where is None or where == []:
            return self
        raise NotImplementedError

    def drop_dim(self):
        return self

    def dot(self, other):
        raise NotImplementedError

    def transpose(self):
        return SparseBlock(self.matrix.transpose(),
                           self.y_shape, self.x_shape)

    def __neg__(self):
        return SparseBlock(-self.matrix, self.x_shape, self.y_shape)

    def bin_op(self, other, op_type):
        scalar = _as_scalar(other)
        if scalar is not None:
            if scalar == 0:
                if op_type in (TensorOp.ADD, TensorOp.SUB):
                    return self
                if op_type == TensorOp.RSUB:
                    return -self
                if op_type in (TensorOp.MUL, TensorOp.RDIV):
                    return Tensor(0)
                raise ZeroDivisionError
            if op_type in (TensorOp.MUL, TensorOp.DIV):
                matrix = self.matrix * scalar if op_type == TensorOp.MUL \
                    else self.matrix / scalar
                return SparseBlock(matrix, self.x_shape, self.y_shape)
            raise InvalidOperation

        if op_type not in _ADDITIVE:
            raise InvalidOperation

        if isinstance(other, SparseBlock):
            if (self.x_shape, self.y_shape) != (other.x_shape, other.y_shape):
                raise InvalidOperation
            a = self.matrix.tocoo()
            b = other.matrix.tocoo()
            a_sign = -1 if op_type == TensorOp.RSUB else 1
            b_sign = -1 if op_type == TensorOp.SUB else 1
            return SparseBlock(
                sp.coo_matrix((
                    np.concatenate([a_sign * a.data, b_sign * b.data]),
                    (np.concatenate([a.row, b.row]),
                     np.concatenate([a.col, b.col]))), shape=a.shape),
                self.x_shape, self.y_shape)

        if isinstance(other, GenericTensor):
            other = Tensor(other.toarray(), x_dim=other.x_dim,
                           y_dim=other.y_dim)

        if isinstance(other, Tensor) and other.n == 0 and other.dim == 0:
            # a dense block: the result is dense
            values = np.broadcast_to(other.toarray(),
                                     self.x_shape + self.y_shape)
            return Tensor(op_type.value(self.toarray(), values),
                          x_dim=self.x_dim, y_dim=self.y_dim)

        raise ValueError
