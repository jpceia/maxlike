import abc
import numpy as np
from random import getrandbits
from array import array
from six import with_metaclass
from functools import partial
from enum import Enum


TENSOR_DTYPE = np.float64


class InvalidOperation(Exception):
    """Exception to indicate an invalid operation for Tensors"""


def set_dtype(s):  # pragma: no cover
    global TENSOR_DTYPE
    if isinstance(s, str):
        s = s.lower()
        mapping = {
            "int32": np.int32,
            "int64": np.int64,
            "float32": np.float32,
            "float64": np.float64
        }
        TENSOR_DTYPE = mapping[s]
    else:
        TENSOR_DTYPE = s


def compose_mappings(map1, map2):
    return array('b', [
        -1 if f < 0 else map1[f]
        for k, f in enumerate(map2)
    ])


def arr_swapaxes(arr, q, p, mapping):
    for k, f in enumerate(mapping):
        if f >= 0:
            arr = arr.swapaxes(q + k, p + f)
    return arr


def arr_take_diag(arr, q, p, mapping):
    for k, f in enumerate(mapping):
        if f < 0:
            continue
        arr = np.diagonal(arr, 0, q + k, p + f)  # np.diagonal
        arr = arr.swapaxes(-1, p + f - 1)
        arr = np.expand_dims(arr, k)
    return arr


def arr_expand_diag(arr, q, p, mapping):
    idx = np.ones(arr.ndim, dtype=np.int32)
    for k, f in enumerate(mapping):
        if f < 0:
            continue
        n = arr.shape[p + f]
        idx[q + k] = n
        idx[p + f] = n
        arr = arr * np.eye(n).reshape(idx)
        idx[q + k] = 1
        idx[p + f] = 1
    return arr


def arr_expand_cross_diag(arr, p, map1, map2):
    idx = np.ones(arr.ndim, dtype=np.int32)
    for i, j in zip(map1, map2):
        if i == j:
            continue
        n = arr.shape[p + i]
        idx[p + i] = n
        idx[p + j] = n
        arr = arr * np.broadcast_to(np.eye(n), idx)
        idx[p + i] = 1
        idx[p + j] = 1
    return arr


def arr_swapaxes_cross(arr, p, map1, map2):
    idx = np.ones(arr.ndim, dtype=np.int32)
    for k, f in enumerate(map2):
        for l, g in enumerate(map1):
            if g == f:
                n = arr.shape[k]
                idx[l] = n
                idx[p + k] = n
                arr = arr * np.broadcast_to(np.eye(n), idx)
                idx[l] = 1
                idx[p + k] = 1
                break
        else:
            arr = arr.swapaxes(p + k, p + f)
    return arr


class TensorOp(Enum):
    ID = partial(lambda x: x)
    RSUB = partial(lambda x, y: np.subtract(y, x))
    RDIV = partial(lambda x, y: np.divide(y, x))
    ADD = np.add
    SUB = np.subtract
    MUL = np.multiply
    DIV = np.divide
    NEG = np.negative


class BaseTensor(object, with_metaclass(abc.ABCMeta)):

    __slots__ = ['x_dim', 'y_dim', 'n', 'dim']

    def __init__(self, x_dim=0, y_dim=0, n=0, dim=0):
        assert x_dim >= 0
        assert y_dim >= 0
        assert n >= 0
        assert dim >= 0
        object.__setattr__(self, 'x_dim', x_dim)
        object.__setattr__(self, 'y_dim', y_dim)
        object.__setattr__(self, 'n', n)
        object.__setattr__(self, 'dim', dim)

    @abc.abstractmethod
    def toarray(self):
        """
        Converts a (Base)Tensor to a numpy ndarray
        """

    @abc.abstractmethod
    def sum(self, sum_val=True, sum_dim=True):
        """
        Collapses the value/dim part of the (Base)Tensor
        """

    @abc.abstractmethod
    def expand(self, where, newsize, dim=False):
        """
        Expands the value/dim part of the (Base)Tensor
        to specified number of dimensions
        """

    @abc.abstractmethod
    def flip(self, where, dim=False):
        """
        Flips a (Base)Tensor for specified dimensions on the value/dim space
        """

    @abc.abstractmethod
    def transpose(self):
        """
        Transposes the 'x_dim' and 'y_dim' spaces
        """

    @abc.abstractmethod
    def drop_dim(self):
        """
        Converts the 'dim' extra-dimensional space to a 'value' space
        """

    @abc.abstractmethod
    def dot(self, other):
        """
        Dot product between (Base)Tensors
        """

    @abc.abstractmethod
    def __neg__(self):
        """
        Applies the negation to (Base)Tensor values
        """

    @abc.abstractmethod
    def bin_op(self, op_type):
        """
        Binary operations between (Base)Tensors
        """


    def __str__(self):
        return "shape: [{}, {}, {}, {}]".format(
            self.x_dim, self.y_dim, self.n, self.dim)

    def __add__(self, other):
        return self.bin_op(other, TensorOp.ADD)

    def __radd__(self, other):
        return self.bin_op(other, TensorOp.ADD)

    def __sub__(self, other):
        return self.bin_op(other, TensorOp.SUB)

    def __rsub__(self, other):
        return self.bin_op(other, TensorOp.RSUB)

    def __mul__(self, other):
        return self.bin_op(other, TensorOp.MUL)

    def __rmul__(self, other):
        return self.bin_op(other, TensorOp.MUL)

    def __truediv__(self, other):
        return self.bin_op(other, TensorOp.DIV)

    def __rtruediv__(self, other):
        return self.bin_op(other, TensorOp.RDIV)

    def __setattr__(self, *ignored):
        raise NotImplementedError

    def __delattr__(self, *ignored):
        raise NotImplementedError


class GenericTensor(BaseTensor):

    __slots__ = ['elements']

    def __init__(self, x_dim=0, y_dim=0, n=0, dim=0, elements=None):
        super(GenericTensor, self).__init__(x_dim, y_dim, n, dim)
        # all of the elements need to have the same shape
        if elements is None:
            elements = ()

        object.__setattr__(self, 'elements', tuple(elements))

    def toarray(self):
        return sum([el.toarray() if isinstance(el, Tensor) else el
                    for el in self.elements])

    def sum(self, sum_val=True, sum_dim=True):
        # considers the impact of broadcasting
        if not sum_val and not sum_dim:
            return self

        if sum_val is False:
            idx = []
        elif sum_val is True:
            idx = np.arange(self.n).tolist()
        else:
            if isinstance(sum_val, int):
                idx = [sum_val]
            if isinstance(sum_val, (tuple, list)):
                idx = list(sum_val)
            else:
                raise ValueError

            if len(idx) == self.n:
                sum_val = True

        if sum_dim is True:
            idx += (self.n + np.arange(self.dim)).tolist()

        shape_n = np.ones(len(idx), dtype=np.int)
        el_sizes = np.empty(len(self.elements), dtype=np.int)
        for k, el in enumerate(self.elements):
            p = el.x_dim + el.y_dim
            shape = np.array(el.values.shape)[(p + np.array(idx)).tolist()]
            shape_n = np.maximum(shape_n, shape)
            el_sizes[k] = np.array(shape).prod()

        size = shape_n.prod()

        if sum_val is True:
            values = 0
            for el, el_size in zip(self.elements, el_sizes):
                idx = []
                idx += [None] * (self.x_dim - el.x_dim) + [slice(None)] * el.x_dim
                idx += [None] * (self.y_dim - el.y_dim) + [slice(None)] * el.y_dim
                idx += [...]
                values = values + el.sum(sum_val, sum_dim).\
                    values[tuple(idx)] * (size / el_size)

            dim = 0 if sum_dim is True else self.dim
            return Tensor(values, x_dim=self.x_dim, y_dim=self.y_dim, dim=dim)

        # else:
        new_elements = [el.sum(False, sum_dim) * (size / el_size)
                        for el, el_size in zip(self.elements, el_sizes)]
        return GenericTensor(self.x_dim, self.y_dim, self.n, 0, new_elements)

    def expand(self, where, newsize, dim=False):
        new_n = self.n if dim else newsize
        new_dim = newsize if dim else self.dim
        return GenericTensor(
            self.x_dim, self.y_dim, new_n, new_dim,
            [el.expand(where, newsize, dim) for el in self.elements])

    def flip(self, where, dim=False):
        return GenericTensor(
            self.x_dim, self.y_dim, self.n, self.dim,
            [el.flip(where, dim) for el in self.elements])

    def transpose(self):
        return GenericTensor(
            self.y_dim, self.x_dim, self.n, self.dim,
            [el.transpose() for el in self.elements])

    def drop_dim(self):
        return GenericTensor(
            self.x_dim, self.y_dim, self.n + self.dim, 0,
            [el.drop_dim() for el in self.elements])

    def dot(self, other):
        if isinstance(other, Tensor):
            gt = GenericTensor()
            for el in self.elements:
                gt = gt + el.dot(other)
            return gt

        if isinstance(other, GenericTensor):
            gt = GenericTensor()
            for el in other.elements:
                gt = gt + self.dot(el)
            return gt

        raise ValueError

    def dot_left(self, other):
        if isinstance(other, Tensor):
            gt = GenericTensor()
            for el in self.elements:
                gt = gt + el.dot_left(other)
            return gt

        if isinstance(other, GenericTensor):
            gt = GenericTensor()
            for el in other.elements:
                gt = gt + self.dot_left(el)
            return gt

        raise ValueError

    def dot_right(self, other):
        if isinstance(other, Tensor):
            gt = GenericTensor()
            for el in self.elements:
                gt = gt + el.dot_right(other)
            return gt

        if isinstance(other, GenericTensor):
            gt = GenericTensor()
            for el in other.elements:
                gt = gt + self.dot_right(el)
            return gt

        raise ValueError

    def __neg__(self):
        return GenericTensor(
            self.x_dim, self.y_dim, self.n, self.dim,
            [-el for el in self.elements])

    def bin_op(self, other, op_type):
        # Scalar
        # Array
        #  -> are converted to tensor
        if isinstance(other, (int, float, np.ndarray)):
            x_dim = self.x_dim
            y_dim = self.y_dim
            n = self.n
            dim = self.dim

        else:
            x_dim = max(self.x_dim, other.x_dim)
            y_dim = max(self.y_dim, other.y_dim)
            n = max(self.n, other.n)
            dim = max(self.dim, other.dim)

        # Tensor
        if isinstance(other, (int, float, np.ndarray, Tensor)):
            new_elements = list(self.elements)
            if op_type in [TensorOp.ADD, TensorOp.SUB]:
                for k, el in enumerate(new_elements):
                    new_el = el.bin_op(other, op_type)
                    if isinstance(new_el, Tensor):
                        new_elements[k] = new_el
                        break
                else:
                    # other isn't coherent with any of the current elements
                    if op_type == TensorOp.ADD:
                        new_elements.append(other)
                    elif op_type == TensorOp.SUB:
                        new_elements.append(-other)
                    else:
                        raise InvalidOperation

            elif op_type == TensorOp.RSUB:
                new_elements = [-el for el in new_elements]
                for k, el in enumerate(new_elements):
                    new_el = el.bin_op(other, TensorOp.ADD)
                    if isinstance(new_el, Tensor):
                        new_elements[k] = new_el
                        break
                else:
                    new_elements.append(other)

            elif op_type in [TensorOp.MUL, TensorOp.DIV]:
                for k, el in enumerate(new_elements):
                    new_elements[k] = el.bin_op(other, op_type)
            else:
                raise InvalidOperation

            return GenericTensor(x_dim, y_dim, n, dim, new_elements)

        # GenericTensor
        if isinstance(other, GenericTensor):
            if op_type in [TensorOp.ADD, TensorOp.SUB]:
                gt = GenericTensor(self.x_dim, self.y_dim, self.n, self.dim,
                                   self.elements)
                for el in other.elements:
                    gt = gt.bin_op(el, op_type)
                return gt
            elif op_type == TensorOp.MUL:
                gt = GenericTensor(x_dim, y_dim, n, dim)
                for el in other.elements:
                    gt = gt + self * el
                return gt
            else:
                raise InvalidOperation

        raise ValueError

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        name = ufunc.__name__
        try:
            op_type = {
                'add': TensorOp.ADD,
                'subtract': TensorOp.RSUB,
                'multiply': TensorOp.MUL,
                'true_divide': TensorOp.RDIV,
                'divide': TensorOp.RDIV,
            }[name]
        except:
            raise InvalidOperation('Invalid ufunc:', name)
        return self.bin_op(inputs[0], op_type)


class Tensor(BaseTensor):

    __slots__ = ['values', 'x_map', 'y_map']

    def __init__(self, values, x_dim=0, y_dim=0, dim=0,
                 x_map=None, y_map=None, dtype=None):

        if isinstance(values, Tensor):
            assert values.x_dim == 0
            assert values.y_dim == 0
            dim = values.dim

        if dtype is None:
            dtype = TENSOR_DTYPE

        object.__setattr__(self, 'values', np.asarray(values, dtype=dtype))
        self.values.flags["WRITEABLE"] = False

        super(Tensor, self).__init__(
            x_dim, y_dim, self.values.ndim - x_dim - y_dim - dim, dim)

        if x_map:
            if not isinstance(x_map, array):
                raise ValueError("Invalid mapping")
        else:
            x_map = array('b')

        if y_map:
            if not isinstance(y_map, array):
                raise ValueError("Invalid mapping")
        else:
            y_map = array('b')

        object.__setattr__(self, 'x_map', x_map)
        object.__setattr__(self, 'y_map', y_map)

    def toarray(self):
        arr = self.values
        p = self.x_dim + self.y_dim
        arr = arr_expand_diag(arr,          0, p, self.x_map)
        arr = arr_expand_diag(arr, self.x_dim, p, self.y_map)
        return arr

    def sum(self, sum_val=True, sum_dim=True):
        val = self.values
        p = self.x_dim + self.y_dim

        if sum_val is False:
            if sum_dim is True:
                val = val.sum(tuple(p + self.n + np.arange(self.dim)))
            return Tensor(val, self.x_dim, self.y_dim, 0,
                          self.x_map, self.y_map)

        if sum_val is True:
            sum_val = np.arange(self.n).tolist()

        if isinstance(sum_val, int):
            sum_val = [sum_val]

        assert isinstance(sum_val, (list, tuple))

        # swapping for x
        for k, f in enumerate(self.x_map):
            if f in sum_val:
                val = val.swapaxes(k, p + f)

        # swapping for y + cross_map setup
        cross_map = {}
        for k, f in enumerate(self.y_map):
            if f not in sum_val:
                continue
            if f in self.x_map:
                l = self.x_map.index(f)
                cross_map[l] = k
            else:
                val = val.swapaxes(self.x_dim + k, p + f)

        idx = p + np.array(sum_val)
        if sum_dim is True:
            idx = idx.tolist() + (p + self.n + np.arange(self.dim)).tolist()
            dim = 0
        else:
            dim = self.dim

        val = np.asarray(val.sum(tuple(idx)))

        # applying cross_map
        for k, f in cross_map.items():
            idx = [None] * val.ndim
            idx[k] = slice(None)
            idx[self.x_dim + f] = slice(None)
            val = val * np.eye(val.shape[k])[tuple(idx)]

        return Tensor(val, self.x_dim, self.y_dim, dim)

    def expand(self, where, newsize, dim=False):
        """
        Applied either to N or to Dim, accordingly with the 'dim' flag
        """
        assert max(where or [-1]) < newsize

        if self.values.ndim == 0:
            if float(self.values) == 0:
                return Tensor(0)

        if dim is True:
            assert len(where) == self.dim
            idx = [slice(None)] * (self.x_dim + self.y_dim + self.n)
            idx += [slice(None) if k in where else None
                    for k in range(newsize)]
            rng = np.arange(self.values.ndim)
            pn = self.x_dim + self.y_dim + self.n
            rng[pn:] = rng[pn:][np.argsort(where)]
            return Tensor(
                self.values.transpose(rng)[tuple(idx)],
                x_dim=self.x_dim,
                y_dim=self.y_dim,
                dim=newsize,
                x_map=self.x_map,
                y_map=self.y_map)

        else:
            assert len(where) == self.n
            idx = [slice(None)] * (self.x_dim + self.y_dim)
            idx += [slice(None) if k in where else None
                    for k in range(newsize)]
            idx += [slice(None)] * self.dim
            p = self.x_dim + self.y_dim
            pn = p + self.n
            rng = np.arange(self.values.ndim)
            rng[p:pn] = rng[p:pn][np.argsort(where)]
            where = array('b', where)
            x_map = compose_mappings(where, self.x_map)
            y_map = compose_mappings(where, self.y_map)
            return Tensor(
                self.values.transpose(rng)[tuple(idx)],
                x_dim=self.x_dim,
                y_dim=self.y_dim,
                dim=self.dim,
                x_map=x_map,
                y_map=y_map)

    def flip(self, where, dim=False):
        if where is None or where == []:
            return self

        if isinstance(where, int):
            where = [where]

        val = self.values
        p = self.x_dim + self.y_dim

        if dim is True:
            # assert max(where) < self.dim
            p += self.n

        else:
            # assert max(where) < self.n
            mappings = array('b')
            mappings += self.x_map
            mappings += self.y_map
            for k in where:
                if k in mappings:
                    raise NotImplementedError

        for k in where:
            val = np.flip(val, p + k)

        return Tensor(
            val,
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            dim=self.dim,
            x_map=self.x_map,
            y_map=self.y_map)

    def transpose(self):
        val = self.values
        if (self.x_dim > 0) & (self.y_dim > 0):
            for k in range(0, self.x_dim):
                val = np.moveaxis(val, self.x_dim, k)
        return Tensor(val, self.y_dim,  self.x_dim, self.dim,
                      self.y_map, self.x_map)

    def drop_dim(self):
        return Tensor(
            self.values, x_dim=self.x_dim, y_dim=self.y_dim, dim=0,
            x_map=self.x_map, y_map=self.y_map)

    def dot(self, other):

        if self.values.ndim == 0:
            if self.values == 0:
                return self

        assert (self.dim == 0) | (other.dim == 0) | (self.dim == other.dim)

        if (other.x_dim > 0) & (other.y_dim > 0):

            if self.x_dim > 0:
                return self.dot_right(other)

            if self.y_dim > 0:
                return self.transpose().dot_right(other.transpose()).transpose()

            raise ValueError

        if other.x_dim > 0:
            return self.dot_left(other)

        # from here other.x_dim == 0

        if other.y_dim > 0:
            return self.transpose().dot_left(other.transpose()).transpose()

        # from here other.y_dim == 0

        if other.n == self.x_dim:
            return self.dot_left(other)

        if (other.n > 0) & (other.n == self.y_dim):
            return self.transpose().dot_left(other)

        if (other.values.ndim == 0) and (other.values == 0):
            return 0

        raise ValueError

    def dot_left(self, other):
        """
        Matrix - Vector inner product

        X - N E (other)
        N Y M E (self)

        product
        X N - - E
        - N Y M E
          *
        """
        dim = max(self.dim, other.dim)

        if (self.x_dim != other.n) & (self.y_dim == other.n):
            return self.transpose().dot_left(other)

        if isinstance(other, GenericTensor):
            return GenericTensor(
                other.x_dim, self.y_dim, self.n, dim,
                [self.dot_left(el) for el in other.elements])

        p = self.x_dim + self.y_dim

        l_idx = [slice(None)] * (other.x_dim + other.n) + \
                [None] * (self.y_dim + self.n) + \
                [slice(None)] * other.dim + \
                [None] * (dim - other.dim)
        r_idx = [None] * other.x_dim + [...] + [None] * (dim - self.dim)

        val = self.values
        val = arr_swapaxes(val, 0, p, self.x_map)
        val = np.multiply(other.values[tuple(l_idx)], val[tuple(r_idx)])
        val = arr_swapaxes(val, other.x_dim, other.x_dim + p, self.x_map)
        if self.x_map:
            x_map = compose_mappings(self.x_map, other.x_map)
        else:
            x_map = array('b')
            val = arr_swapaxes(val, 0, other.x_dim, other.x_map)

        val = val.sum(tuple(other.x_dim + np.arange(other.n)))
        return Tensor(val, x_dim=other.x_dim, y_dim=self.y_dim, dim=dim,
                      x_map=x_map, y_map=self.y_map)

    def dot_right(self, other):
        """
        Vector - Matrix  inner product

        X Y N E (other)
        N - M E (self)

        product
        X Y N - E
        - - N M E
            *
        """

        try:
            if other.values.ndim == 0 and other.values == 0:
                return 0
        except:
            pass

        dim = max(self.dim, other.dim)

        assert self.y_dim == 0
        assert self.x_dim == other.n

        if isinstance(other, GenericTensor):
            return GenericTensor(
                other.x_dim, other.y_dim, self.n, dim,
                [self.dot_right(el) for el in other.elements])

        p = other.x_dim + other.y_dim

        # to do, broadcast on dim
        l_idx = [slice(None)] * (p + other.n) + [None] * self.n + \
                [slice(None)] * other.dim + [None] * (dim - other.dim)
        r_idx = [None] * p + [slice(None)] * (self.x_dim + self.n) + \
                [None] * (dim - self.dim)

        val = self.values
        val = arr_swapaxes(val, 0, self.x_dim, self.x_map)
        val = other.values[tuple(l_idx)] * val[tuple(r_idx)]
        val = arr_swapaxes(val, p, p + self.x_dim, self.x_map)
        if not self.x_map:
            val = arr_swapaxes(val, 0, p, other.x_map)
            val = arr_swapaxes_cross(val, other.x_dim,                      # NOT COVERED
                                     other.x_map, other.y_map)

        x_map = compose_mappings(self.x_map, other.x_map)
        y_map = compose_mappings(self.x_map, other.y_map)
        val = val.sum(tuple(p + np.arange(other.n)))
        return Tensor(val,
                      x_dim=other.x_dim, y_dim=other.y_dim, dim=dim,
                      x_map=x_map, y_map=y_map)

    def __neg__(self):
        return Tensor(
            -self.values,
            x_dim=self.x_dim, y_dim=self.y_dim, dim=self.dim,
            x_map=self.x_map, y_map=self.y_map)

    def __pow__(self, a):
        return Tensor(
            np.power(self.values, a),
            x_dim=self.x_dim, y_dim=self.y_dim, dim=self.dim,
            x_map=self.x_map, y_map=self.y_map)

    def bin_op(self, other, op_type):

        if isinstance(other, GenericTensor):
            if op_type in [TensorOp.ADD, TensorOp.MUL]:
                new_op = op_type
            elif op_type == TensorOp.SUB:
                new_op = TensorOp.RSUB
            else:
                raise InvalidOperation
            return other.bin_op(self, new_op)

        is_scalar = False
        if np.isscalar(other):
            is_scalar = True
        elif isinstance(other, np.ndarray):
            if other.ndim == 0:
                is_scalar = True
            else:
                other = Tensor(other)
        elif isinstance(other, Tensor):
            if other.values.ndim == 0:
                is_scalar = True
                other = other.values
        else:
            raise ValueError

        if is_scalar:
            # special case: 0
            if other == 0:
                if op_type in [TensorOp.ADD, TensorOp.SUB]:
                    return self
                elif op_type in [TensorOp.RSUB]:
                    return -self
                elif op_type in [TensorOp.MUL, TensorOp.RDIV]:
                    return Tensor(0)
                elif op_type == TensorOp.DIV:
                    raise ZeroDivisionError
                else:
                    raise InvalidOperation

            # special case: 1
            elif other == 1:
                if op_type in [TensorOp.MUL, TensorOp.DIV]:
                    return self

            if op_type in [TensorOp.ADD, TensorOp.SUB, TensorOp.RSUB]:
                return Tensor(
                    op_type.value(self.toarray(), other),
                    x_dim=self.x_dim, y_dim=self.y_dim, dim=self.dim)

            elif op_type in [TensorOp.MUL, TensorOp.DIV, TensorOp.RDIV]:
                return Tensor(
                    op_type.value(self.values, other),
                    x_dim=self.x_dim, y_dim=self.y_dim, dim=self.dim,
                    x_map=self.x_map, y_map=self.y_map)
            else:
                raise InvalidOperation

        if self.values.ndim == 0:
            return Tensor(
                op_type.value(self.values, other.values),
                other.x_dim, other.y_dim, other.dim, other.x_map, other.y_map)

        # l_idx and r_idx could be defined
        assert (self.n == 0) | (other.n == 0) | (self.n == other.n)
        n = max(self.n, other.n)

        # set dim
        assert (self.dim == 0) | (other.dim == 0) | \
               (self.dim == other.dim)
        dim = max(self.dim, other.dim)

        # set x_dim
        assert (self.x_dim == 0) | (other.x_dim == 0) | \
               (self.x_dim == other.x_dim)
        x_dim = max(self.x_dim, other.x_dim)

        # set y_dim
        assert (self.y_dim == 0) | (other.y_dim == 0) | \
               (self.y_dim == other.y_dim)
        y_dim = max(self.y_dim, other.y_dim)

        l_values = self.values
        r_values = other.values

        l_idx = self.__reshape_idx(x_dim, y_dim, n, dim)
        r_idx = other.__reshape_idx(x_dim, y_dim, n, dim)

        # --------------------------------------------------------------------
        # Addition and Subtraction
        # --------------------------------------------------------------------
        if op_type in [TensorOp.ADD, TensorOp.SUB, TensorOp.RSUB]:
            if (self.x_map == other.x_map) & \
                    (self.y_map == other.y_map):
                return Tensor(
                    op_type.value(l_values[l_idx], r_values[r_idx]),
                    x_dim=x_dim, y_dim=y_dim, dim=dim,
                    x_map=self.x_map, y_map=self.y_map)

            # there are other cases where we can do merging
            # when the shape of one of the arrays 'contains' the other

            if op_type == TensorOp.ADD:
                elements = [self, other]
            elif op_type == TensorOp.SUB:
                elements = [self, -other]
            elif op_type == TensorOp.RSUB:
                elements = [-self, other]
            else:
                raise ValueError

            return GenericTensor(x_dim, y_dim, n, dim, elements)

        # --------------------------------------------------------------------
        # Multiplication and Division
        # --------------------------------------------------------------------
        p = self.x_dim + self.y_dim

        if self.x_map:
            x_map = self.x_map
            l_values = arr_expand_cross_diag(
                l_values, p, self.x_map, other.x_map)
            if not other.x_map and other.x_dim:
                r_values = arr_take_diag(r_values, 0, other.x_dim + other.y_dim,
                                         self.x_map)
        else:
            x_map = other.x_map
            if x_map and self.x_dim:
                l_values = arr_take_diag(l_values, 0, p, other.x_map)

        if self.y_map:
            y_map = self.y_map
            l_values = arr_expand_cross_diag(
                l_values, p, self.y_map, other.y_map)
            if not other.y_map and other.y_dim:
                r_values = arr_take_diag(r_values, other.x_dim, other.x_dim + other.y_dim,
                                         self.y_map)
        else:
            y_map = other.y_map
            if self.y_dim:
                l_values = arr_take_diag(l_values, self.x_dim, p, other.y_map)

        return Tensor(
            op_type.value(l_values[l_idx], r_values[r_idx]),
            x_dim=x_dim, y_dim=y_dim, dim=dim,
            x_map=x_map, y_map=y_map)

    def __float__(self):
        if self.values.ndim > 0:
            raise ValueError
        return float(self.values)

    def __array__(self, *args, **kwargs):
        return self.values

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        name = ufunc.__name__

        if name in (
            "exp", "log", "gamma", "square", "gammaln",
            "expit", "logit", "tanh", "arctan"):
            if (self.x_dim > 0) | (self.y_dim > 0):
                raise InvalidOperation(
                    "Unary ufuncs only supported for Tensor" +
                    "for x_dim == y_dim == 0")
            return Tensor(ufunc(self.values), x_dim=0, y_dim=0, dim=self.dim)

        if name in ("maximum", "minimum"):
            if (self.x_dim > 0) | (self.y_dim > 0):
                raise InvalidOperation(
                    "{} only supported for Tensor".format(name) +
                    "for x_dim == y_dim == 0")
            return Tensor(ufunc(self.values, inputs[1]),
                          x_dim=0, y_dim=0, dim=self.dim)

        try:
            op_type = {
                'add': TensorOp.ADD,
                'subtract': TensorOp.RSUB,
                'multiply': TensorOp.MUL,
                'true_divide': TensorOp.RDIV,
                'divide': TensorOp.RDIV,
            }[name]
        except:
            raise InvalidOperation('Invalid ufunc:', name)
        return self.bin_op(inputs[0], op_type)

    def __reshape_idx(self, x_dim, y_dim, n, dim):
        idx = []
        if self.x_dim > 0:
            # assert self.x_dim == x_dim
            idx += [slice(None)] * x_dim
        elif self.x_dim == 0:
            idx += [None] * x_dim
        if self.y_dim > 0:
            # assert self.y_dim == y_dim
            idx += [slice(None)] * y_dim
        elif self.y_dim == 0:
            idx += [None] * y_dim
        if self.n > 0:
            # assert self.n == n
            idx += [slice(None)] * n
        elif self.n == 0:
            idx += [None] * n
        if self.dim > 0:
            # assert self.dim == dim
            idx += [slice(None)] * dim
        elif self.dim == 0:
            idx += [None] * dim

        return tuple(idx)
