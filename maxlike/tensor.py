import abc
import numpy as np
from array import array
from six import with_metaclass


class BaseTensor(with_metaclass(abc.ABCMeta)):

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

    def __neg__(self):
        return self * (-1.0)

    def __add__(self, other):
        return self._bin_op(other, "add")

    def __radd__(self, other):
        return self._bin_op(other, "add")

    def __sub__(self, other):
        return self._bin_op(other, "sub")

    def __rsub__(self, other):
        return self._bin_op(other, "rsub")

    def __mul__(self, other):
        return self._bin_op(other, "mul")

    def __rmul__(self, other):
        return self._bin_op(other, "mul")

    def __div__(self, other):
        return self._bin_op(other, "div")

    def __truediv__(self, other):
        return self._bin_op(other, "div")

    def __rdiv__(self, other):
        return self._bin_op(other, "rdiv")

    def __rtruediv__(self, other):
        return self._bin_op(other, "rdiv")


class GenericTensor(BaseTensor):
    def __init__(self, p1=0, p2=0, n=0, dim=0, elements=None):
        super(GenericTensor, self).__init__(p1, p2, n, dim)
        # all of the elements need to have the same shape
        if elements:
            self.elements = elements
        else:
            self.elements = []

    def sum(self, sum_val=True, sum_dim=True):
        # consider the impact of broadcasting
        if not sum_val and not sum_dim:
            return self.copy()
        i0 = 0
        i1 = self.n + self.dim
        if not sum_val:
            i0 = self.n
        if not sum_dim:
            i1 = self.dim

        shape_n = [0] * (i1 - i0)
        sizes = []
        for el in self.elements:
            p = el.p1 + el.p2
            el_size = 1
            for i, k in enumerate(el.values.shape[p+i0:p+i1]):
                el_size *= k
                shape_n[i] = max(shape_n[i], k)
            sizes.append(el_size)

        size = 1
        for k in shape_n:
            size *= k

        values = 0
        for el, el_size in zip(self.elements, sizes):
            values = values + el.sum(sum_val, sum_dim).values * (size / el_size)

        dim = 0 if sum_dim is True else self.dim
        return Tensor(values, p1=self.p1, p2=self.p2, dim=dim)

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

    def copy(self):
        return GenericTensor(
            self.p1, self.p2, self.n, self.dim,
            [el.copy() for el in self.elements])

    def shape(self):
        return "(p1:%d, p2:%d, n:%d, v:%d)" % \
            (self.p1, self.p2, self.n, self.dim)

    def __len__(self):
        return len(self.elements)

    def _bin_op(self, other, op_type):
        # Scalar
        # Array
        #  -> are converted to tensor
        if isinstance(other, (int, float, np.ndarray)):
            return self._bin_op(Tensor(other), op_type)

        p1 = max(self.p1, other.p1)
        p2 = max(self.p2, other.p2)
        n = max(self.n, other.n)
        dim = max(self.dim, other.dim)

        # Tensor
        if isinstance(other, Tensor):
            new_elements = self.elements[:]
            if op_type in ["add", "sub"]:
                for k, el in enumerate(new_elements):
                    new_el = el._bin_op(other, op_type)
                    if isinstance(new_el, Tensor):
                        new_elements[k] = new_el
                        break
                else:
                    # other isn't coherent with any of the current elements
                    new_elements.append(other)
            elif op_type in ["mul", "div", "rdiv"]:
                for k, el in enumerate(new_elements):
                    new_elements[k] = el._bin_op(other, op_type)
            else:
                raise ValueError

            return GenericTensor(p1, p2, n, dim, new_elements)

        # GenericTensor
        if isinstance(other, GenericTensor):
            if op_type in ["add", "sub"]:
                gt = self.copy()
                for el in other.elements:
                    gt = gt._bin_op(el, op_type)
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

    def sum(self, sum_val=True, sum_dim=True):
        t = self.copy()
        p = self.p1 + self.p2
        cross_map = {}

        if not sum_val:
            if sum_dim:
                t.values = t.values.sum(tuple(
                    p + self.n + np.arange(self.dim)))
                t.dim = 0
            return t

        if len(self.p1_mapping) > 0:
            for k, f in enumerate(self.p1_mapping):
                if f >= 0:
                    t.values = t.values.swapaxes(k, p + f)
        if len(self.p2_mapping) > 0:
            for l, f in enumerate(self.p2_mapping):
                if f >= 0:
                    if len(self.p1_mapping) > 0 and f in self.p1_mapping:
                        k = self.p1_mapping.index(f)
                        cross_map[k] = l
                    else:
                        t.values = t.values.swapaxes(self.p1 + l, p + f)

        if sum_dim is True:
            idx = tuple(p + np.arange(self.n + self.dim))
            t.dim = 0
        else:
            idx = tuple(p + np.arange(self.n))

        t.values = t.values.sum(idx).transpose()
        t.p1_mapping = array('b')
        t.p2_mapping = array('b')
        t.n = 0

        for k, l in cross_map.items():
            idx = np.zeros(t.values.ndim, dtype=np.bool)
            idx[k] = True
            idx[self.p1 + l] = True
            idx = [slice(None) if x else None for x in idx]
            t.values = t.values * np.eye(t.values.shape[self.p1 + l])[idx]

        return t

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
            return Tensor(
                self.values[idx],
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
            p1_mapping = array('b')
            p2_mapping = array('b')
            if len(self.p1_mapping) > 0:
                for k in self.p1_mapping:
                    if k < 0:
                        p1_mapping.append(-1)
                    else:
                        p1_mapping.append(xmap[k])
            if len(self.p2_mapping) > 0:
                for k in self.p2_mapping:
                    if k < 0:
                        p2_mapping.append(-1)
                    else:
                        p2_mapping.append(xmap[k])
            return Tensor(
                self.values[idx],
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
            assert max(xmap) < self.dim
            p += self.n

        else:
            assert max(xmap) < self.n
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
        t = self.copy()
        if (self.p1 != 0) & (self.p2 != 0):
            for k in range(0, self.p1):
                t.values = np.moveaxis(t.values, self.p1, k)
        t.p1 = self.p2
        t.p2 = self.p1
        t.p1_mapping = self.p2_mapping
        t.p2_mapping = self.p1_mapping
        return t

    def drop_dim(self):
        return Tensor(
            self.values, p1=self.p1, p2=self.p2, dim=0,
            p1_mapping=self.p1_mapping, p2_mapping=self.p2_mapping)

    def dot(self, other):

        if self.values.ndim == 0:
            if self.values == 0:
                return self

        assert (self.dim == 0) | (other.dim == 0) | (self.dim == other.dim)
        dim = max(self.dim, other.dim)

        if (other.p1 > 0) & (other.p2 > 0):

            if self.p1 > 0:

                # P1 P2 N E (other)
                # N  -  M E (self)
                #
                # product
                # P1 P2 N - E
                # -  -  N M E
                #       x

                assert self.p2 == 0
                assert self.p1 == other.n

                if isinstance(other, GenericTensor):
                    return GenericTensor(
                        other.p1, other.p2, self.n, dim,
                        [self.dot(el) for el in other.elements])

                p = other.p1 + other.p2
                dim = max(self.dim, other.dim)

                # to do, broadcast on dim
                l_idx = [slice(None)] * (p + other.n) + [None] * self.n + \
                        [slice(None)] * other.dim + [None] * (dim - other.dim)
                r_idx = [None] * p + [Ellipsis] + [None] * (dim - self.dim)

                p1_mapping = array('b')
                p2_mapping = array('b')
                val = self.values.copy()
                if len(self.p1_mapping) > 0:
                    for k, f in enumerate(self.p1_mapping):
                        if f >= 0:
                            val = val.swapaxes(k, self.p1 + f)
                val = other.values[l_idx] * val[r_idx]

                if len(self.p1_mapping) > 0:
                    for k, f in enumerate(self.p1_mapping):
                        if f >= 0:
                            val = val.swapaxes(p + k, p + self.n + f)

                    if len(other.p1_mapping) > 0:
                        p1_mapping = array('b')
                        for k in other.p1_mapping:
                            if k < 0:
                                p1_mapping.append(-1)
                            else:
                                p1_mapping.append(self.p1_mapping[k])

                    if len(other.p2_mapping) > 0:
                        p2_mapping = array('b')
                        for k in other.p2_mapping:
                            if k < 0:
                                p2_mapping.append(-1)
                            else:
                                p2_mapping.append(self.p1_mapping[k])
                else:
                    if len(other.p1_mapping) > 0:
                        for k, f in enumerate(other.p1_mapping):
                            if f >= 0:
                                val = val.swapaxes(k, p + f)
                    if len(other.p2_mapping) > 0:
                        for k, f in enumerate(other.p2_mapping):
                            if f is not None:
                                if len(other.p1_mapping) > 0 and \
                                        f in other.p1_mapping:
                                    # overlap
                                    idx = np.zeros(val.ndim, dtype=np.bool)
                                    idx[other.p1_mapping.index(f)] = True
                                    idx[other.p1 + k] = True
                                    idx = [slice(None) if x else None
                                           for x in idx]
                                    val = val * np.eye(val.shape[k])[idx]
                                else:
                                    # no overlap
                                    val = val.swapaxes(other.p1 + k, p + f)

                val = val.sum(tuple(p + np.arange(other.n)))
                return Tensor(val, p1=other.p1, p2=other.p2, dim=dim,
                              p1_mapping=p1_mapping,
                              p2_mapping=p2_mapping)

            elif self.p2 > 0:
                return self.transpose().dot(other.transpose()).transpose()
            else:
                raise ValueError

        elif (other.p1 > 0) | ((other.p1 == 0) & (other.p2 == 0) & (other.n == self.p1)):

            # P1 -  N E (other)
            # N  P2 M E (self)
            #
            # product
            # P1 N -  - E
            # -  N P2 M E
            #    x

            assert self.p1 == other.n

            if isinstance(other, GenericTensor):
                return GenericTensor(
                    other.p1, self.p2, self.n, dim,
                    [self.dot(el) for el in other.elements])

            p = self.p1 + self.p2

            l_idx = [slice(None)] * (other.p1 + other.n) + \
                    [None] * (self.p2 + self.n) + [slice(None)] * other.dim + \
                    [None] * (dim - other.dim)
            r_idx = [None] * other.p1 + [Ellipsis] + [None] * (dim - self.dim)

            p1_mapping = array('b')
            val = self.values.copy()

            if len(self.p1_mapping) > 0:
                for k, f in enumerate(self.p1_mapping):
                    if f >= 0:
                        val = val.swapaxes(k, p + f)
            val = other.values[l_idx] * val[r_idx]

            if len(self.p1_mapping) > 0:
                for k, f in enumerate(self.p1_mapping):
                    if f >= 0:
                        val = val.swapaxes(other.p1 + k, other.p1 + p + f)
                if len(other.p1_mapping) > 0:
                    p1_mapping = array('b')
                    for f in other.p1_mapping:
                        if f < 0:
                            p1_mapping.append(-1)
                        else:
                            p1_mapping.append(self.p1_mapping[f])
            else:
                if len(other.p1_mapping) > 0:
                    for k, f in enumerate(other.p1_mapping):
                        if f >= 0:
                            val = val.swapaxes(k, other.p1 + f)

            val = val.sum(tuple(other.p1 + np.arange(other.n)))
            return Tensor(val, p1=other.p1, p2=self.p2, dim=dim,
                          p1_mapping=p1_mapping,
                          p2_mapping=self.p2_mapping)

        elif other.p2 > 0:
            return self.transpose().dot(other.transpose()).transpose()

        elif other.values == 0:
            return 0
        raise ValueError

    def copy(self):
        return Tensor(self.values, p1=self.p1, p2=self.p2, dim=self.dim,
                      p1_mapping=self.p1_mapping, p2_mapping=self.p2_mapping)

    def _bin_op(self, other, op_type):

        if isinstance(other, GenericTensor):
            p1 = max(self.p1, other.p1)
            p2 = max(self.p2, other.p2)
            n = max(self.n, other.n)
            dim = max(self.dim, other.dim)

            new_elements = other.elements[:]
            if op_type in ["add", "sub"]:
                for k, el in enumerate(new_elements):
                    new_el = self._bin_op(el, op_type)
                    if isinstance(new_el, Tensor):
                        new_elements[k] = new_el
                        break
                else:
                    # other isn't coherent with any of the current elements
                    new_elements.append(other)
            elif op_type in ["mul", "div"]:
                for k, el in enumerate(new_elements):
                    new_elements[k] = self._bin_op(el, op_type)
            else:
                raise ValueError

            return GenericTensor(p1, p2, n, dim, new_elements)

        if isinstance(other, (int, float, np.ndarray)):
            return self._bin_op(Tensor(other), op_type)

        scalar_op = False
        if other.values.ndim == 0:
            scalar_op = True
        elif self.values.ndim == 0:
            scalar_op = True

        if scalar_op:
            if other.values.ndim == 0:
                t = self.copy()
            else:
                t = other.copy()
            if op_type == "add":
                t.values = np.asarray(self.values + other.values)
            elif op_type == "sub":
                t.values = np.asarray(self.values - other.values)
            elif op_type == "rsub":
                t.values = np.asarray(other.values - self.values)
            elif op_type == "mul":
                t.values = np.asarray(self.values * other.values)
            elif op_type == "div":
                t.values = np.asarray(self.values / other.values)
            elif op_type == "rdiv":
                t.values = np.asarray(other.values / self.values)
            else:
                raise NotImplementedError
            return t

        values = None

        assert self.n == other.n
        n = self.n

        # set new_e
        assert (self.dim == 0) | (other.dim == 0) | (self.dim == other.dim)
        dim = max(self.dim, other.dim)

        # set new_p1
        p1 = max(self.p1, other.p1)
        assert (self.p1 == 0) | (other.p1 == 0) | (self.p1 == other.p1)

        # set new_p2
        p2 = max(self.p2, other.p2)
        assert (self.p2 == 0) | (other.p2 == 0) | (self.p2 == other.p2)

        if self.p1 == other.p1:
            if self.p1_mapping == other.p1_mapping:
                p1_mapping = self.p1_mapping
            else:
                if op_type == "add":  # synthetic sum
                    return GenericTensor(
                        p1, p2, n, dim, [self.copy(), other.copy()])
                elif op_type == "sub":
                    return GenericTensor(
                        p1, p2, n, dim, [self.copy(), -other.copy()])
                elif op_type in ["mul", "div"]:
                    if values is None:
                        values = self.values.copy()
                    p1_mapping = self.p1_mapping
                    p = self.p1 + self.p2
                    for fs, fo in zip(p1_mapping, other.p1_mapping):
                        if fs != fo:
                            idx = np.zeros(values.ndim, dtype=np.bool)
                            idx[p + fs] = True
                            idx[p + fo] = True
                            idx = [slice(None) if i else None for i in idx]
                            values = values * np.eye(values.shape[p + fs])[idx]
                else:
                    raise NotImplementedError
        elif other.p1 == 0:
            p1_mapping = self.p1_mapping
        elif self.p1 == 0:
            p1_mapping = other.p1_mapping
        else:
            raise ValueError

        if self.p2 == other.p2:
            if self.p2_mapping == other.p2_mapping:
                p2_mapping = self.p2_mapping
            else:
                if op_type == "add":  # synthetic sum
                    return GenericTensor(
                        p1, p2, n, dim, [self.copy(), other.copy()])
                elif op_type == "sub":
                    return GenericTensor(
                        p1, p2, n, dim, [self.copy(), -other.copy()])
                elif op_type in ["mul", "div"]:
                    if values is None:
                        values = self.values.copy()
                    p2_mapping = self.p2_mapping
                    p = self.p1 + self.p2
                    for fs, fo in zip(p2_mapping, other.p2_mapping):
                        if fs != fo:
                            idx = np.zeros(values.ndim, dtype=np.bool)
                            idx[p + fs] = True
                            idx[p + fo] = True
                            idx = [slice(None) if i else None for i in idx]
                            values = values * np.eye(values.shape[p + fs])[idx]
                else:
                    raise NotImplementedError

        elif other.p2 == 0:
            p2_mapping = self.p2_mapping
        elif self.p2 == 0:
            p2_mapping = other.p2_mapping
        else:
            raise ValueError

        # adjust the values
        l_idx = self.__reshape_idx(p1, p2, n, dim)
        r_idx = other.__reshape_idx(p1, p2, n, dim)

        if values is None:
            values = self.values.copy()

        if op_type == "add":
            values = values[l_idx] + other.values[r_idx]
        elif op_type == "sub":
            values = values[l_idx] - other.values[r_idx]
        elif op_type == "mul":
            values = values[l_idx] * other.values[r_idx]
        elif op_type == "div":
            values = values[l_idx] / other.values[r_idx]
        else:
            raise NotImplementedError

        return Tensor(values, p1=p1, p2=p2, dim=dim,
                      p1_mapping=p1_mapping, p2_mapping=p2_mapping)

    def shape(self):
        return "(p1:%d, p2:%d, n:%d, v:%d)" % \
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
        #return self.values.__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def __reshape_idx(self, p1, p2, n, dim):
        idx = []
        if self.p1 > 0:
            p1 = max(1, p1)
            assert self.p1 == p1
            idx += [slice(None)] * p1
        elif (self.p1 == 0) & (p1 > 0):
            idx += [None] * p1
        if self.p2 > 0:
            p2 = max(1, p2)
            assert self.p2 == p2
            idx += [slice(None)] * p2
        elif (self.p2 == 0) & (p2 > 0):
            idx += [None] * p2
        if self.n > 0:
            assert self.n == n
            idx += [slice(None)] * n
        elif (self.n == 0) & (n > 0):
            idx += [None] * n
        if self.dim > 0:
            assert self.dim == dim
            idx += [slice(None)] * dim
        elif (self.dim == 0) & (dim > 0):
            idx += [None] * dim
        return idx


def grad_tensor(values, params, i=0, p1_mapping=None, dim=0):
    p1 = np.asarray(params[i]).ndim
    if p1_mapping is None:
        idx = [Ellipsis]
    else:
        idx = [None] * p1 + [Ellipsis]
    return Tensor(values[idx], p1=p1, dim=dim, p1_mapping=p1_mapping)


def hess_tensor(values, params, i=0, j=0,
                p1_mapping=None, p2_mapping=None, dim=0):
    p1 = np.asarray(params[i]).ndim
    p2 = np.asarray(params[j]).ndim
    idx = [slice(None) if p1_mapping is None else None] * p1
    idx += [slice(None) if p2_mapping is None else None] * p2
    idx += [Ellipsis]
    return Tensor(values[idx], p1=p1, p2=p2, dim=dim,
                  p1_mapping=p1_mapping, p2_mapping=p2_mapping)
