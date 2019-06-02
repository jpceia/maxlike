import abc
import numpy as np
from .ctensor import *
from random import getrandbits
from array import array
from six import with_metaclass
from functools import partial
from enum import Enum


TENSOR_DTYPE = np.float64


class InvalidOperation(Exception):
    """Exception to indicate an invalid operation for Tensors"""


def set_dtype(s): # pragma: no cover
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

    __slots__ = ['p1', 'p2', 'n', 'dim', 'hash']

    def __init__(self, p1=0, p2=0, n=0, dim=0):
        assert p1 >= 0
        assert p2 >= 0
        assert n >= 0
        assert dim >= 0
        object.__setattr__(self, 'p1', p1)
        object.__setattr__(self, 'p2', p2)
        object.__setattr__(self, 'n', n)
        object.__setattr__(self, 'dim', dim)
        object.__setattr__(self, 'hash', getrandbits(128))

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
    def expand(self, xmap, newsize, dim=False):
        """
        Expands the value/dim part of the (Base)Tensor
        to specified number of dimensions
        """

    @abc.abstractmethod
    def flip(self, xmap, dim=False):
        """
        Flips a (Base)Tensor for specified dimensions on the value/dim space
        """

    @abc.abstractmethod
    def transpose(self):
        """
        Transposes the 'p1' and 'p2' spaces
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

    def __hash__(self):
        return self.hash

    def __str__(self):
        return "shape: [{}, {}, {}, {}]".format(
            self.p1, self.p2, self.n, self.dim)

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

    def __div__(self, other):
        return self.bin_op(other, TensorOp.DIV)

    def __truediv__(self, other):
        return self.bin_op(other, TensorOp.DIV)

    def __rdiv__(self, other):
        return self.bin_op(other, TensorOp.RDIV)

    def __rtruediv__(self, other):
        return self.bin_op(other, TensorOp.RDIV)

    def __setattr__(self, *ignored):
        raise NotImplementedError

    def __delattr__(self, *ignored):
        raise NotImplementedError


class GenericTensor(BaseTensor):

    __slots__ = ['elements']

    def __init__(self, p1=0, p2=0, n=0, dim=0, elements=None):
        super(GenericTensor, self).__init__(p1, p2, n, dim)
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

        i0 = 0
        i1 = self.n + self.dim

        if not sum_val:
            i0 = self.n

        if not sum_dim:
            i1 = self.dim

        shape_n = np.ones(i1 - i0, dtype=np.int)
        el_sizes = np.empty(len(self.elements), dtype=np.int)
        for k, el in enumerate(self.elements):
            p = el.p1 + el.p2
            shape = el.values.shape[p + i0:p + i1]
            shape_n = np.maximum(shape_n, shape)
            el_sizes[k] = np.array(shape).prod()

        size = shape_n.prod()

        if sum_val:
            values = 0
            for el, el_size in zip(self.elements, el_sizes):
                idx = []
                idx += [None] * (self.p1 - el.p1) + [slice(None)] * el.p1
                idx += [None] * (self.p2 - el.p2) + [slice(None)] * el.p2
                idx += [...]
                values = values + el.sum(sum_val, sum_dim).\
                    values[tuple(idx)] * (size / el_size)

            dim = 0 if sum_dim is True else self.dim
            return Tensor(values, p1=self.p1, p2=self.p2, dim=dim)

        #else:
        new_elements = [el.sum(False, sum_dim) * (size / el_size)
                        for el, el_size in zip(self.elements, el_sizes)]
        return GenericTensor(self.p1, self.p2, self.n, 0, new_elements)

    def expand(self, xmap, newsize, dim=False):
        new_n = self.n if dim else newsize
        new_dim = newsize if dim else self.dim
        return GenericTensor(
            self.p1, self.p2, new_n, new_dim,
            [el.expand(xmap, newsize, dim) for el in self.elements])

    def flip(self, xmap, dim=False):
        return GenericTensor(
            self.p1, self.p2, self.n, self.dim,
            [el.flip(xmap, dim) for el in self.elements])

    def transpose(self):
        return GenericTensor(
            self.p2, self.p1, self.n, self.dim,
            [el.transpose() for el in self.elements])

    def drop_dim(self):
        return GenericTensor(
            self.p1, self.p2, self.n + self.dim, 0,
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
            self.p1, self.p2, self.n, self.dim,
            [-el for el in self.elements])

    def bin_op(self, other, op_type):
        # Scalar
        # Array
        #  -> are converted to tensor
        if isinstance(other, (int, float, np.ndarray)):
            p1 = self.p1
            p2 = self.p2
            n = self.n
            dim = self.dim

        else:
            p1 = max(self.p1, other.p1)
            p2 = max(self.p2, other.p2)
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

            return GenericTensor(p1, p2, n, dim, new_elements)

        # GenericTensor
        if isinstance(other, GenericTensor):
            if op_type in [TensorOp.ADD, TensorOp.SUB]:
                gt = GenericTensor(self.p1, self.p2, self.n, self.dim,
                                   self.elements)
                for el in other.elements:
                    gt = gt.bin_op(el, op_type)
                return gt
            elif op_type == TensorOp.MUL:
                gt = GenericTensor(p1, p2, n, dim)
                for el in other.elements:
                    gt = gt + self * el
                return gt
            else:
                raise InvalidOperation

        raise ValueError


class Tensor(BaseTensor):

    __slots__ = ['values', 'p1_mapping', 'p2_mapping']

    def __init__(self, values, p1=0, p2=0, dim=0,
                 p1_mapping=None, p2_mapping=None, dtype=None):

        if dtype is None:
            dtype = TENSOR_DTYPE

        object.__setattr__(self, 'values', np.asarray(values, dtype=dtype))
        self.values.flags["WRITEABLE"] = False

        super(Tensor, self).__init__(
            p1, p2, self.values.ndim - p1 - p2 - dim, dim)

        if p1_mapping:
            if not isinstance(p1_mapping, array):
                raise ValueError("Invalid mapping")
        else:
            p1_mapping = array('b')

        if p2_mapping:
            if not isinstance(p2_mapping, array):
                raise ValueError("Invalid mapping")
        else:
            p2_mapping = array('b')

        object.__setattr__(self, 'p1_mapping', p1_mapping)
        object.__setattr__(self, 'p2_mapping', p2_mapping)

    def toarray(self):
        arr = self.values
        p = self.p1 + self.p2
        arr = arr_expand_diag(arr,       0, p, self.p1_mapping)
        arr = arr_expand_diag(arr, self.p1, p, self.p2_mapping)
        return arr

    def sum(self, sum_val=True, sum_dim=True):
        val = self.values
        p = self.p1 + self.p2
        cross_map = {}

        if not sum_val:
            if sum_dim:
                val = val.sum(tuple(p + self.n + np.arange(self.dim)))
            return Tensor(val, self.p1, self.p2, 0,
                          self.p1_mapping, self.p2_mapping)

        val = arr_swapaxes(val, 0, p, self.p1_mapping)
        for k, f in enumerate(self.p2_mapping):
            if f < 0:
                continue
            if f in self.p1_mapping:
                l = self.p1_mapping.index(f)
                cross_map[l] = k
            else:
                val = val.swapaxes(self.p1 + k, p + f)

        if sum_dim is True:
            idx = tuple(p + np.arange(self.n + self.dim))
            dim = 0
        else:
            idx = tuple(p + np.arange(self.n))
            dim = self.dim

        val = np.asarray(val.sum(idx)).transpose()

        for k, f in cross_map.items():
            idx = [None] * val.ndim
            idx[k] = slice(None)
            idx[self.p1 + f] = slice(None)
            val = val * np.eye(val.shape[self.p1 + f])[tuple(idx)]

        return Tensor(val, self.p1, self.p2, dim)

    def expand(self, xmap, newsize, dim=False):
        """
        Applied either to N or to Dim, accordingly with the 'dim' flag
        """
        assert max(xmap or [-1]) < newsize

        if self.values.ndim == 0:
            if float(self.values) == 0:
                return Tensor(0)

        if dim is True:
            assert len(xmap) == self.dim
            idx = [slice(None)] * (self.p1 + self.p2 + self.n)
            idx += [slice(None) if k in xmap else None
                    for k in range(newsize)]
            rng = np.arange(self.values.ndim)
            pn = self.p1 + self.p2 + self.n
            rng[pn:] = rng[pn:][np.argsort(xmap)]
            return Tensor(
                self.values.transpose(rng)[tuple(idx)],
                p1=self.p1,
                p2=self.p2,
                dim=newsize,
                p1_mapping=self.p1_mapping,
                p2_mapping=self.p2_mapping)

        else:
            assert len(xmap) == self.n
            idx = [slice(None)] * (self.p1 + self.p2)
            idx += [slice(None) if k in xmap else None
                    for k in range(newsize)]
            idx += [slice(None)] * self.dim
            p = self.p1 + self.p2
            pn = p + self.n
            rng = np.arange(self.values.ndim)
            rng[p:pn] = rng[p:pn][np.argsort(xmap)]
            xmap = array('b', xmap)
            p1_mapping = compose_mappings(xmap, self.p1_mapping)
            p2_mapping = compose_mappings(xmap, self.p2_mapping)
            return Tensor(
                self.values.transpose(rng)[tuple(idx)],
                p1=self.p1,
                p2=self.p2,
                dim=self.dim,
                p1_mapping=p1_mapping,
                p2_mapping=p2_mapping)

    def flip(self, xmap, dim=False):
        if xmap is None or xmap == []:
            return self

        if isinstance(xmap, int):
            xmap = [xmap]

        val = self.values
        p = self.p1 + self.p2

        if dim is True:
            # assert max(xmap) < self.dim
            p += self.n

        else:
            # assert max(xmap) < self.n
            mappings = array('b')
            mappings += self.p1_mapping
            mappings += self.p2_mapping
            for k in xmap:
                if k in mappings:
                    raise NotImplementedError

        for k in xmap:
            val = np.flip(val, p + k)

        return Tensor(
            val,
            p1=self.p1,
            p2=self.p2,
            dim=self.dim,
            p1_mapping=self.p1_mapping,
            p2_mapping=self.p2_mapping)

    def transpose(self):
        val = self.values
        if (self.p1 > 0) & (self.p2 > 0):
            for k in range(0, self.p1):
                val = np.moveaxis(val, self.p1, k)
        return Tensor(val, self.p2,  self.p1, self.dim,
                      self.p2_mapping, self.p1_mapping)

    def drop_dim(self):
        return Tensor(
            self.values, p1=self.p1, p2=self.p2, dim=0,
            p1_mapping=self.p1_mapping, p2_mapping=self.p2_mapping)

    def dot(self, other):

        if self.values.ndim == 0:
            if self.values == 0:
                return self

        assert (self.dim == 0) | (other.dim == 0) | (self.dim == other.dim)

        if (other.p1 > 0) & (other.p2 > 0):

            if self.p1 > 0:
                return self.dot_right(other)
            elif self.p2 > 0:
                return self.transpose().dot_right(other.transpose()).transpose()
            else:
                raise ValueError

        elif other.p1 > 0:
            return self.dot_left(other)

        # from here other.p1 == 0

        elif other.p2 > 0:
            return self.transpose().dot_left(other.transpose()).transpose()

        # from here other.p2 == 0

        elif other.n == self.p1:
            return self.dot_left(other)

        elif (other.n > 0) & (other.n == self.p2):
            return self.transpose().dot_left(other)

        elif other.values == 0:
            return 0
        
        raise ValueError

    def dot_left(self, other):
        """
        Matrix - Vector inner product

        P1 -  N E (other)
        N  P2 M E (self)
        
        product
        P1 N -  - E
        -  N P2 M E
           x
        """
        dim = max(self.dim, other.dim)

        if (self.p1 != other.n) & (self.p2 == other.n):
            return self.transpose().dot_left(other)

        if isinstance(other, GenericTensor):
            return GenericTensor(
                other.p1, self.p2, self.n, dim,
                [self.dot_left(el) for el in other.elements])

        p = self.p1 + self.p2

        l_idx = [slice(None)] * (other.p1 + other.n) + \
                [None] * (self.p2 + self.n) + [slice(None)] * other.dim + \
                [None] * (dim - other.dim)
        r_idx = [None] * other.p1 + [...] + [None] * (dim - self.dim)

        val = self.values
        val = arr_swapaxes(val, 0, p, self.p1_mapping)
        val = np.multiply(other.values[tuple(l_idx)], val[tuple(r_idx)])
        val = arr_swapaxes(val, other.p1, other.p1 + p, self.p1_mapping)
        if self.p1_mapping:
            p1_mapping = compose_mappings(self.p1_mapping, other.p1_mapping)
        else:
            p1_mapping = array('b')
            val = arr_swapaxes(val, 0, other.p1, other.p1_mapping)

        val = val.sum(tuple(other.p1 + np.arange(other.n)))
        return Tensor(val, p1=other.p1, p2=self.p2, dim=dim,
                      p1_mapping=p1_mapping,
                      p2_mapping=self.p2_mapping)

    def dot_right(self, other):
        """
        Vector - Matrix  inner product

        P1 P2 N E (other)
        N  -  M E (self)

        product
        P1 P2 N - E
        -  -  N M E
              x
        """

        try:
            if other.values.ndim == 0 and other.values == 0:
                return 0
        except:
            pass

        dim = max(self.dim, other.dim)

        assert self.p2 == 0
        assert self.p1 == other.n

        if isinstance(other, GenericTensor):
            return GenericTensor(
                other.p1, other.p2, self.n, dim,
                [self.dot_right(el) for el in other.elements])

        p = other.p1 + other.p2

        # to do, broadcast on dim
        l_idx = [slice(None)] * (p + other.n) + [None] * self.n + \
                [slice(None)] * other.dim + [None] * (dim - other.dim)
        r_idx = [None] * p + [slice(None)] * (self.p1 + self.n) + \
                [None] * (dim - self.dim)

        val = self.values
        val = arr_swapaxes(val, 0, self.p1, self.p1_mapping)
        val = other.values[tuple(l_idx)] * val[tuple(r_idx)]
        val = arr_swapaxes(val, p, p + self.p1, self.p1_mapping)
        if not self.p1_mapping:
            val = arr_swapaxes(val, 0, p, other.p1_mapping)
            val = arr_swapaxes_cross(val, other.p1,
                  other.p1_mapping, other.p2_mapping)

        p1_mapping = compose_mappings(self.p1_mapping, other.p1_mapping)
        p2_mapping = compose_mappings(self.p1_mapping, other.p2_mapping)
        val = val.sum(tuple(p + np.arange(other.n)))
        return Tensor(val, p1=other.p1, p2=other.p2, dim=dim,
                      p1_mapping=p1_mapping,
                      p2_mapping=p2_mapping)

    def __neg__(self):
        return Tensor(
            -self.values,
            p1=self.p1, p2=self.p2, dim=self.dim,
            p1_mapping=self.p1_mapping, p2_mapping=self.p2_mapping)

    def __pow__(self, a):
        return Tensor(
            np.power(self.values, a),
            p1=self.p1, p2=self.p2, dim=self.dim,
            p1_mapping=self.p1_mapping, p2_mapping=self.p2_mapping)

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
                    p1=self.p1, p2=self.p2, dim=self.dim)

            elif op_type in [TensorOp.MUL, TensorOp.DIV, TensorOp.RDIV]:
                return Tensor(
                    op_type.value(self.values, other),
                    p1=self.p1, p2=self.p2, dim=self.dim,
                    p1_mapping=self.p1_mapping,
                    p2_mapping=self.p2_mapping)
            else:
                raise InvalidOperation

        if self.values.ndim == 0:
            return Tensor(
                op_type.value(self.values, other.values),
                other.p1, other.p2, other.dim, other.p1_mapping, other.p2_mapping)

        # l_idx and r_idx could be defined
        assert (self.n == 0) | (other.n == 0) | (self.n == other.n)
        n = max(self.n, other.n)

        # set dim
        assert (self.dim == 0) | (other.dim == 0) | (self.dim == other.dim)
        dim = max(self.dim, other.dim)

        # set p1
        assert (self.p1 == 0) | (other.p1 == 0) | (self.p1 == other.p1)
        p1 = max(self.p1, other.p1)

        # set p2
        assert (self.p2 == 0) | (other.p2 == 0) | (self.p2 == other.p2)
        p2 = max(self.p2, other.p2)

        l_values = self.values
        r_values = other.values

        l_idx = self.__reshape_idx(p1, p2, n, dim)
        r_idx = other.__reshape_idx(p1, p2, n, dim)

        # --------------------------------------------------------------------
        # Addition and Subtraction
        # --------------------------------------------------------------------
        if op_type in [TensorOp.ADD, TensorOp.SUB, TensorOp.RSUB]:
            if (self.p1_mapping == other.p1_mapping) & \
                    (self.p2_mapping == other.p2_mapping):
                return Tensor(
                    op_type.value(l_values[l_idx], r_values[r_idx]),
                    p1=p1, p2=p2, dim=dim,
                    p1_mapping=self.p1_mapping,
                    p2_mapping=self.p2_mapping)

            # there are other cases where we can do merging
            # when the shape of one of the arrays 'contains' the other

            if op_type == TensorOp.ADD:
                elements = [self, other]
            elif op_type ==  TensorOp.SUB:
                elements = [self, -other]
            elif op_type == TensorOp.RSUB:
                elements = [-self, other]
            else:
                raise ValueError

            return GenericTensor(p1, p2, n, dim, elements)

        # --------------------------------------------------------------------
        # Multiplication and Division
        # --------------------------------------------------------------------
        p = self.p1 + self.p2

        if self.p1_mapping:
            p1_mapping = self.p1_mapping
            l_values = arr_expand_cross_diag(
                l_values, p, self.p1_mapping, other.p1_mapping)
            if not other.p1_mapping and other.p1:
                r_values = arr_take_diag(r_values, 0, other.p1 + other.p2,
                                         self.p1_mapping)
        else:
            p1_mapping = other.p1_mapping
            if p1_mapping and self.p1:
                l_values = arr_take_diag(l_values, 0, p, other.p1_mapping)

        if self.p2_mapping:
            p2_mapping = self.p2_mapping
            l_values = arr_expand_cross_diag(
                l_values, p, self.p2_mapping, other.p2_mapping)
            if not other.p2_mapping and other.p2: 
                r_values = arr_take_diag(r_values, other.p1, other.p1 + other.p2,
                                         self.p2_mapping)
        else:
            p2_mapping = other.p2_mapping
            if self.p2:
                l_values = arr_take_diag(l_values, self.p1, p, other.p2_mapping)

        return Tensor(
            op_type.value(l_values[l_idx], r_values[r_idx]),
            p1=p1, p2=p2, dim=dim,
            p1_mapping=p1_mapping,
            p2_mapping=p2_mapping)

    def __float__(self):
        if self.values.ndim > 0:
            raise ValueError
        return float(self.values)

    def __array__(self, *args, **kwargs):
        return self.values

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        name = ufunc.__name__

        if name in (
            "exp", "log", "gamma", #"square",
            "expit", "logit", "tanh", "arctan"):
            if (self.p1 > 0) | (self.p2 > 0):
                raise InvalidOperation(
                    "Unary ufuncs only supported for Tensor" +
                    "for p1 == p2 == 0")
            return Tensor(ufunc(self.values), p1=0, p2=0, dim=self.dim)

        if name in ("maximum", "minimum"):
            if (self.p1 > 0) | (self.p2 > 0):
                raise InvalidOperation(
                    "{} only supported for Tensor".format(name) +
                    "for p1 == p2 == 0")
            return Tensor(ufunc(self.values, inputs[1]),
                          p1=0, p2=0, dim=self.dim)

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

    def __reshape_idx(self, p1, p2, n, dim):
        idx = []
        if self.p1 > 0:
            # assert self.p1 == p1
            idx += [slice(None)] * p1
        elif self.p1 == 0:
            idx += [None] * p1
        if self.p2 > 0:
            # assert self.p2 == p2
            idx += [slice(None)] * p2
        elif self.p2 == 0:
            idx += [None] * p2
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
