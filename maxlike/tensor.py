import abc
import numpy as np
from array import array
from six import with_metaclass


def _last_diag(arr, axis1, axis2):
    return np.diagonal(arr, axis1=axis1, axis2=axis2). \
        swapaxes(-1, axis2 - 1)


class BaseTensor(with_metaclass(abc.ABCMeta)):
    lambda_op = {
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'rsub': lambda x, y: y - x,
        'mul': lambda x, y: x * y,
        'div': lambda x, y: x / y,
        'rdiv': lambda x, y: y / x,
        'neg': lambda x: -x,
    }

    @abc.abstractmethod
    def __init__(self, p1=0, p2=0, n=0, dim=0):
        assert p1 >= 0
        assert p2 >= 0
        assert n >= 0
        assert dim >= 0
        self.p1 = p1
        self.p2 = p2
        self.n = n
        self.dim = dim

    @abc.abstractmethod
    def toarray(self):
        pass

    @abc.abstractmethod
    def sum(self, sum_val=True, sum_dim=True):
        pass

    @abc.abstractmethod
    def expand(self, xmap, newsize, dim=False):
        pass

    @abc.abstractmethod
    def flip(self, xmap, dim=False):
        pass

    @abc.abstractmethod
    def transpose(self):
        pass

    @abc.abstractmethod
    def drop_dim(self):
        pass

    @abc.abstractmethod
    def dot(self, other):
        pass

    @abc.abstractmethod
    def copy(self):
        pass

    @abc.abstractmethod
    def shape(self):
        pass

    @abc.abstractmethod
    def __neg__(self):
        pass

    @abc.abstractmethod
    def bin_op(self, op_type):
        pass

    def __add__(self, other):
        return self.bin_op(other, "add")

    def __radd__(self, other):
        return self.bin_op(other, "add")

    def __sub__(self, other):
        return self.bin_op(other, "sub")

    def __rsub__(self, other):
        return self.bin_op(other, "rsub")

    def __mul__(self, other):
        return self.bin_op(other, "mul")

    def __rmul__(self, other):
        return self.bin_op(other, "mul")

    def __div__(self, other):
        return self.bin_op(other, "div")

    def __truediv__(self, other):
        return self.bin_op(other, "div")

    def __rdiv__(self, other):
        return self.bin_op(other, "rdiv")

    def __rtruediv__(self, other):
        return self.bin_op(other, "rdiv")


class GenericTensor(BaseTensor):
    def __init__(self, p1=0, p2=0, n=0, dim=0, elements=None):
        super(GenericTensor, self).__init__(p1, p2, n, dim)
        # all of the elements need to have the same shape
        if elements:
            self.elements = elements
        else:
            self.elements = []

    def toarray(self):
        return sum([el.toarray() for el in self.elements])

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

        shape_n = np.zeros(i1 - i0, dtype=np.int)
        sizes = np.zeros(len(self.elements), dtype=np.int)
        for k, el in enumerate(self.elements):
            p = el.p1 + el.p2
            el_size = 1
            for i, u in enumerate(el.values.shape[p + i0:p + i1]):
                el_size *= u
                shape_n[i] = max(shape_n[i], u)
            sizes[k] = el_size

        size = shape_n.prod()

        if sum_val:
            values = 0
            for el, el_size in zip(self.elements, sizes):
                idx = []
                idx += [None] * (self.p1 - el.p1) + [slice(None)] * el.p1
                idx += [None] * (self.p2 - el.p2) + [slice(None)] * el.p2
                idx += [...]
                values = values + \
                    el.sum(sum_val, sum_dim).values[idx] * (size / el_size)

            dim = 0 if sum_dim is True else self.dim
            return Tensor(values, p1=self.p1, p2=self.p2, dim=dim)

        #else:
        new_elements = [el.sum(False, sum_dim) * (size / el_size)
                        for el, el_size in zip(self.elements, sizes)]
        return GenericTensor(self.p1, self.p2, self.n, 0, new_elements)

    def expand(self, xmap, newsize, dim=False):
        new_n = self.n if dim else newsize
        new_dim = newsize if dim else self.dim
        return GenericTensor(
            self.p1, self.p2, new_n, new_dim,
            [el.expand(xmap, newsize, dim) for el in self.elements])

    def flip(self, xmap, dim=False):
        gt = GenericTensor()
        for el in self.elements:
            gt += el.flip(xmap, dim)
        return gt

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
                gt += el.dot(other)
            return gt

        if isinstance(other, GenericTensor):
            gt = GenericTensor()
            for el in other.elements:
                gt += self.dot(el)
            return gt

        raise ValueError

    def dot_left(self, other):
        if isinstance(other, Tensor):
            gt = GenericTensor()
            for el in self.elements:
                gt += el.dot_left(other)
            return gt

        if isinstance(other, GenericTensor):
            gt = GenericTensor()
            for el in other.elements:
                gt += self.dot_left(el)
            return gt

        raise ValueError

    def dot_right(self, other):
        if isinstance(other, Tensor):
            gt = GenericTensor()
            for el in self.elements:
                gt += el.dot_right(other)
            return gt

        if isinstance(other, GenericTensor):
            gt = GenericTensor()
            for el in other.elements:
                gt += self.dot_right(el)
            return gt

        raise ValueError

    def copy(self):
        return GenericTensor(
            self.p1, self.p2, self.n, self.dim,
            [el.copy() for el in self.elements])

    def shape(self):
        return "(p1:%d, p2:%d, n:%d, v:%d)" % \
               (self.p1, self.p2, self.n, self.dim)

    def __len__(self):
        return len(self.elements)

    def __neg__(self):
        return GenericTensor(
            self.p1, self.p2, self.n, self.dim,
            [-el for el in self.elements])

    def bin_op(self, other, op_type):
        # Scalar
        # Array
        #  -> are converted to tensor
        if isinstance(other, (int, float, np.ndarray)):
            return self.bin_op(Tensor(other), op_type)

        p1 = max(self.p1, other.p1)
        p2 = max(self.p2, other.p2)
        n = max(self.n, other.n)
        dim = max(self.dim, other.dim)

        # Tensor
        if isinstance(other, Tensor):
            new_elements = self.elements[:]
            if op_type in ["add", "sub"]:
                for k, el in enumerate(new_elements):
                    new_el = el.bin_op(other, op_type)
                    if isinstance(new_el, Tensor):
                        new_elements[k] = new_el
                        break
                else:
                    # other isn't coherent with any of the current elements
                    if op_type == "add":
                        new_elements.append(other)
                    elif op_type == "sub":
                        new_elements.append(-other)
                    else:
                        raise ValueError

            elif op_type == "rsub":
                new_elements = [-el for el in new_elements]
                for k, el in enumerate(new_elements):
                    new_el = el.bin_op(other, "add")
                    if isinstance(new_el, Tensor):
                        new_elements[k] = new_el
                        break
                else:
                    new_elements.append(other)

            elif op_type in ["mul", "div"]:
                for k, el in enumerate(new_elements):
                    new_elements[k] = el.bin_op(other, op_type)
            else:
                raise ValueError

            return GenericTensor(p1, p2, n, dim, new_elements)

        # GenericTensor
        if isinstance(other, GenericTensor):
            if op_type in ["add", "sub"]:
                gt = self.copy()
                for el in other.elements:
                    gt = gt.bin_op(el, op_type)
                return gt
            elif op_type == "mul":
                gt = GenericTensor(p1, p2, n, dim)
                for el in other.elements:
                    gt += self * el
                return gt
            else:
                raise ValueError

        raise ValueError

    def __getitem__(self, i):
        return self.elements[i]


class Tensor(BaseTensor):
    def __init__(self, values=0, p1=0, p2=0, dim=0,
                 p1_mapping=None, p2_mapping=None):

        self.values = np.asarray(values)
        self.values.flags["WRITEABLE"] = False
        super(Tensor, self).__init__(
            p1, p2, self.values.ndim - p1 - p2 - dim, dim)
        self.p1_mapping = array('b')
        self.p2_mapping = array('b')
        if p1_mapping:
            if p1_mapping is True:
                assert p1 == self.n
                self.p1_mapping = array('b', range(p1))
            elif isinstance(p1_mapping, (list, tuple, range, array)):
                assert len(p1_mapping) == p1
                self.p1_mapping = array('b', p1_mapping)
            else:
                raise ValueError("p1_mapping defined incorrectly")
        if p2_mapping:
            if p2_mapping is True:
                assert p2 == self.n
                self.p2_mapping = array('b', range(p2))
            elif isinstance(p2_mapping, (list, tuple, range, array)):
                assert len(p2_mapping) == p2
                self.p2_mapping = array('b', p2_mapping)
            else:
                raise ValueError("p2_mapping defined incorrectly")

    def toarray(self):
        arr = self.values
        p = self.p1 + self.p2
        for k, f in enumerate(self.p1_mapping):
            if f >= 0:
                idx = [None] * arr.ndim
                idx[k] = slice(None)
                idx[p + f] = slice(None)
                arr = arr * np.eye(arr.shape[p + f])[idx]

        for k, f in enumerate(self.p2_mapping):
            if f >= 0:
                idx = [None] * arr.ndim
                idx[self.p1 + k] = slice(None)
                idx[p + f] = slice(None)
                arr = arr * np.eye(arr.shape[p + f])[idx]
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

        if self.p1_mapping:
            for k, f in enumerate(self.p1_mapping):
                if f >= 0:
                    val = val.swapaxes(k, p + f)
        if self.p2_mapping:
            for k, f in enumerate(self.p2_mapping):
                if f >= 0:
                    if self.p1_mapping and f in self.p1_mapping:
                        k = self.p1_mapping.index(f)
                        cross_map[k] = k
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
            val = val * np.eye(val.shape[self.p1 + f])[idx]

        return Tensor(val, self.p1, self.p2, dim)

    def expand(self, xmap, newsize, dim=False):
        """
        Applied either to N or to Dim, accordingly with the 'dim' flag
        """
        assert max(xmap or [-1]) < newsize

        if self.values.ndim == 0:
            if self.values == 0:
                return Tensor()

        if dim is True:
            assert len(xmap) == self.dim
            idx = [slice(None)] * (self.p1 + self.p2 + self.n)
            idx += [slice(None) if k in xmap else None
                    for k in range(newsize)]
            rng = np.arange(self.values.ndim)
            pn = self.p1 + self.p2 + self.n
            rng[pn:] = rng[pn:][np.argsort(xmap)]
            return Tensor(
                self.values.transpose(rng)[idx],
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
            p1_mapping = array('b')
            p2_mapping = array('b')

            if self.p1_mapping:
                for k in self.p1_mapping:
                    if k < 0:
                        p1_mapping.append(-1)
                    else:
                        p1_mapping.append(xmap[k])

            if self.p2_mapping:
                for k in self.p2_mapping:
                    if k < 0:
                        p2_mapping.append(-1)
                    else:
                        p2_mapping.append(xmap[k])

            return Tensor(
                self.values.transpose(rng)[idx],
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
        if (self.p1 != 0) & (self.p2 != 0):
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

        p1_mapping = array('b')
        val = self.values

        if self.p1_mapping:
            for k, f in enumerate(self.p1_mapping):
                if f >= 0:
                    val = val.swapaxes(k, p + f)
        val = other.values[l_idx] * val[r_idx]

        if self.p1_mapping:
            
            for k, f in enumerate(self.p1_mapping):
                if f >= 0:
                    val = val.swapaxes(other.p1 + k, other.p1 + p + f)
            
            if other.p1_mapping:
                for f in other.p1_mapping:
                    if f < 0:
                        p1_mapping.append(-1)
                    else:
                        p1_mapping.append(self.p1_mapping[f])
        else:
            if other.p1_mapping:
                for k, f in enumerate(other.p1_mapping):
                    if f >= 0:
                        val = val.swapaxes(k, other.p1 + f)

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

        p1_mapping = array('b')
        p2_mapping = array('b')

        val = self.values
        if self.p1_mapping:
            for k, f in enumerate(self.p1_mapping):
                if f >= 0:
                    val = val.swapaxes(k, self.p1 + f)
        val = other.values[l_idx] * val[r_idx]

        if self.p1_mapping:
            for k, f in enumerate(self.p1_mapping):
                if f >= 0:
                    val = val.swapaxes(p + k, p + self.p1 + f)

            if other.p1_mapping:
                for k in other.p1_mapping:
                    if k < 0:
                        p1_mapping.append(-1)
                    else:
                        p1_mapping.append(self.p1_mapping[k])

            if other.p2_mapping:
                for k in other.p2_mapping:
                    if k < 0:
                        p2_mapping.append(-1)
                    else:
                        p2_mapping.append(self.p1_mapping[k])
        else:
            if other.p1_mapping:
                for k, f in enumerate(other.p1_mapping):
                    if f >= 0:
                        val = val.swapaxes(k, p + f)

            if other.p2_mapping:
                for k, f in enumerate(other.p2_mapping):
                    if f is not None:
                        if other.p1_mapping and f in other.p1_mapping:
                            # overlap
                            idx = [None] * val.ndim
                            idx[other.p1_mapping.index(f)] = slice(None)
                            idx[other.p1 + k] = slice(None)
                            val = val * np.eye(val.shape[k])[idx]
                        else:
                            # no overlap
                            val = val.swapaxes(other.p1 + k, p + f)

        val = val.sum(tuple(p + np.arange(other.n)))
        return Tensor(val, p1=other.p1, p2=other.p2, dim=dim,
                      p1_mapping=p1_mapping,
                      p2_mapping=p2_mapping)

    def copy(self):
        return Tensor(self.values, p1=self.p1, p2=self.p2, dim=self.dim,
                      p1_mapping=self.p1_mapping, p2_mapping=self.p2_mapping)

    def __neg__(self):
        return Tensor(
            -self.values, p1=self.p1, p2=self.p2, dim=self.dim,
            p1_mapping=self.p1_mapping, p2_mapping=self.p2_mapping)

    def bin_op(self, other, op_type):

        if isinstance(other, GenericTensor):
            if op_type in ["add", "mul"]:
                new_op = op_type
            elif op_type == "sub":
                new_op = "rsub"
            else:
                raise ValueError("Only +, -, * for T / GT operations")

            return other.bin_op(self, new_op)

        if isinstance(other, (int, float)):
            return Tensor(
                Tensor.lambda_op[op_type](self.values, other),
                p1=self.p1, p2=self.p2, dim=self.dim,
                p1_mapping=self.p1_mapping,
                p2_mapping=self.p2_mapping)

        if isinstance(other, np.ndarray):
            idx = [None] * (self.p1 + self.p2) + [...] + [None] * self.dim
            return Tensor(
                Tensor.lambda_op[op_type](self.values, other[idx]),
                p1=self.p1, p2=self.p2, dim=self.dim,
                p1_mapping=self.p1_mapping,
                p2_mapping=self.p2_mapping)

        if other.values.ndim == 0 or self.values.ndim == 0:
            if other.values.ndim == 0:
                t = self.copy()
            else:
                t = other.copy()
            t.values = np.asarray(Tensor.lambda_op[op_type](
                self.values, other.values))
            return t

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
        # Addition and Multiplication
        # --------------------------------------------------------------------
        if op_type in ["add", "sub", "rsub"]:
            if (self.p1_mapping == other.p1_mapping) and \
                    (self.p2_mapping == other.p2_mapping):
                return Tensor(
                    Tensor.lambda_op[op_type](
                        l_values[l_idx], r_values[r_idx]),
                    p1=p1, p2=p2, dim=dim,
                    p1_mapping=self.p1_mapping,
                    p2_mapping=self.p2_mapping)

            # there are other cases where we can do merging
            # when the shape of one of the arrays 'contains' the other

            if op_type == "add":
                elements = [self.copy(), other.copy()]
            elif op_type == "sub":
                elements = [self.copy(), -other.copy()]
            elif op_type == "rsub":
                elements = [-self.copy(), other.copy()]
            else:
                raise ValueError

            return GenericTensor(p1, p2, n, dim, elements)

        # --------------------------------------------------------------------
        # Multiplication and Division
        # --------------------------------------------------------------------
        p = self.p1 + self.p2

        if self.p1_mapping:
            p1_mapping = self.p1_mapping
            if other.p1_mapping:
                if p1_mapping != other.p1_mapping:
                    for fs, fo in zip(p1_mapping, other.p1_mapping):
                        if fs != fo:
                            idx = [None] * l_values.ndim
                            idx[p + fs] = slice(None)
                            idx[p + fo] = slice(None)
                            l_values = l_values * np.eye(
                                l_values.shape[p + fs])[idx]
            elif other.p1:
                for k, f in enumerate(p1_mapping):
                    if f >= 0:
                        idx = [slice(None)] * r_values.ndim
                        idx[k] = None
                        r_values = _last_diag(
                            r_values, k, other.p1 + other.p2 + f)[idx]
        else:
            p1_mapping = other.p1_mapping
            if p1_mapping and self.p1:
                for k, f in enumerate(p1_mapping):
                    if f >= 0:
                        idx = [slice(None)] * l_values.ndim
                        idx[k] = None
                        l_values = _last_diag(l_values, k, p + f)[idx]

        if self.p2_mapping:
            p2_mapping = self.p2_mapping
            if other.p2_mapping:
                if self.p2_mapping != other.p2_mapping:
                    for fs, fo in zip(p2_mapping, other.p2_mapping):
                        if fs != fo:
                            idx = [None] * r_values.ndim
                            idx[p + fs] = slice(None)
                            idx[p + fo] = slice(None)
                            l_values = l_values * np.eye(
                                l_values.shape[p + fs])[idx]
            elif other.p2:
                for k, f in enumerate(p2_mapping):
                    if f >= 0:
                        idx = [slice(None)] * r_values.ndim
                        idx[self.p1 + k] = None
                        r_values = _last_diag(
                            r_values,
                            other.p1 + k,
                            other.p1 + other.p2 + f)[idx]
        else:
            p2_mapping = other.p2_mapping
            if other.p2_mapping and self.p2:
                for k, f in enumerate(p2_mapping):
                    if f >= 0:
                        idx = [slice(None)] * l_values.ndim
                        idx[self.p1 + k] = None
                        l_values = _last_diag(
                            l_values, axis1=self.p1 + k, axis2=p + f)[idx]

        return Tensor(
            Tensor.lambda_op[op_type](l_values[l_idx], r_values[r_idx]),
            p1=p1, p2=p2, dim=dim,
            p1_mapping=p1_mapping,
            p2_mapping=p2_mapping)

    def shape(self):
        return "(%d|%d|%d|%d)" % \
               (self.p1, self.p2, self.n, self.dim)

    def __str__(self):
        s = str(self.values)
        s += "\nshape: "
        s += str((self.p1, self.p2, self.n, self.dim))
        return s

    def __repr__(self):  # just displays the shape
        return repr(self.values)

    def __array__(self):
        return self.values

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if (self.p1 > 0) | (self.p2 > 0):
            return ufunc(self.values, *inputs)
        values = ufunc(self.values)
        return Tensor(values, p1=0, p2=0, dim=self.dim)

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

        return idx


def grad_tensor(values, params, i=0, p1_mapping=None, dim=0):
    p1 = np.asarray(params[i]).ndim
    if p1_mapping is None:
        idx = [...]
    else:
        idx = [None] * p1 + [...]
    return Tensor(values[idx], p1=p1, dim=dim, p1_mapping=p1_mapping)


def hess_tensor(values, params, i=0, j=0,
                p1_mapping=None, p2_mapping=None, dim=0):
    p1 = np.asarray(params[i]).ndim
    p2 = np.asarray(params[j]).ndim
    idx = [slice(None) if p1_mapping is None else None] * p1
    idx += [slice(None) if p2_mapping is None else None] * p2
    idx += [...]
    return Tensor(values[idx], p1=p1, p2=p2, dim=dim,
                  p1_mapping=p1_mapping, p2_mapping=p2_mapping)
