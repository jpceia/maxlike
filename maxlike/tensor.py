import abc
import numpy as np


class BaseTensor(object):
    __metaclass__ = abc.ABCMeta

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
    def sum(self, dim=True):
        pass

    @abc.abstractmethod
    def expand(self, feat_map, ndim, dim=False):
        pass

    @abc.abstractmethod
    def transpose(self):
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
    def __add__(self, other):
        pass

    @abc.abstractmethod
    def __sub__(self, other):
        pass

    @abc.abstractmethod
    def __mul__(self, other):
        pass

    @abc.abstractmethod
    def __div__(self, other):
        pass

    def __neg__(self):
        return self * (-1.0)

    def __truediv__(self, other):
        return self.__div__(other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other

    def __rmul__(self, other):
        return self * other

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __rtruediv__(self, other):
        return self / other


class GenericTensor(BaseTensor):
    def __init__(self, p1=0, p2=0, n=0, dim=0, elements=None):
        super(GenericTensor, self).__init__(p1, p2, n, dim)
        # all of the elements need to have the same shape
        if elements:
            self.elements = elements
        else:
            self.elements = []

    def sum(self, dim=True):
        return Tensor(
            sum([el.sum(dim).values for el in self.elements]),
            p1=self.p1, p2=self.p2, dim=self.dim)

    def expand(self, feat_map, ndim, dim=False):
        if dim:
            self.dim = ndim
        else:
            self.n = ndim

        self.elements = [el.expand(feat_map, ndim, dim)
                         for el in self.elements]

    def transpose(self):
        self.p1, self.p2 = self.p2, self.p1
        self.elements = [el.transpose() for el in self.elements]

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

    def _add(self, other, sgn):
        # Scalar
        # Array
        #  -> are converted to tensor
        if isinstance(other, (int, float, np.ndarray)):
            return self._add(Tensor(other), sgn)

        p1 = max(self.p1, other.p1)
        p2 = max(self.p2, other.p2)
        n = max(self.n, other.n)
        dim = max(self.dim, other.dim)

        # Tensor
        if isinstance(other, Tensor):
            new_elements = self.elements[:]
            for k, el in enumerate(new_elements):
                new_el = el + sgn * other
                if isinstance(new_el, Tensor):
                    new_elements[k] = new_el
                    break
            else:
                new_elements.append(other)

            return GenericTensor(p1, p2, n, dim, new_elements)

        # GenericTensor
        if isinstance(other, GenericTensor):
            gt = self.copy()
            for el in other.elements:
                gt = gt.__add(el, sgn)
            return gt

        raise ValueError

    def shape(self):
        return "(p1:%d, p2:%d, n:%d, v:%d)" % \
            (self.p1, self.p2, self.n, self.dim)

    def __add__(self, other):
        return self._add(other, 1.0)

    def __sub__(self, other):
        return self._add(other, -1.0)

    def __len__(self):
        return len(self.elements)

    def _scalar_mul(self, a):
        assert isinstance(a, (int, float))
        return GenericTensor(
            self.p1, self.p2, self.n, self.dim,
            [el * a for el in self.elements])

    # multiplication / division just with scalars
    def __mul__(self, other):
        return self._scalar_mul(other)

    def __div__(self, other):
        return self._scalar_mul(1.0 / other)

    def __neg__(self, other):
        return self._scalar_mul(-1.0)


class Tensor(BaseTensor):
    def __init__(self, values=0, p1=0, p2=0, dim=0,
                 p1_mapping=None, p2_mapping=None):

        self.values = np.asarray(values)
        super(Tensor, self).__init__(
            p1, p2, self.values.ndim - p1 - p2 - dim, dim)
        self.p1_mapping = None
        self.p2_mapping = None
        if p1_mapping is not None:
            if p1_mapping is True:
                assert p1 == self.n
                self.p1_mapping = range(self.n)
            elif isinstance(p1_mapping, (list, tuple, range)):
                assert len(p1_mapping) == p1
                self.p1_mapping = p1_mapping
            else:
                raise ValueError("p1_mapping defined incorrectly")
        if p2_mapping is not None:
            if p2_mapping is True:
                assert p2 == self.n
                self.p2_mapping = range(self.n)
            elif isinstance(p2_mapping, (list, tuple, range)):
                assert len(p2_mapping) == p2
                self.p2_mapping = p2_mapping
            else:
                raise ValueError("p2_mapping defined incorrectly")

    def sum(self, val=True, dim=True):
        t = self.copy()
        p = self.p1 + self.p2

        if not val:
            if dim:
                t.values = t.values.sum(tuple(
                    p + self.n + np.arange(self.dim)))
                t.dim = 0
            return t

        if self.p1_mapping is not None:
            for k, f in enumerate(self.p1_mapping):
                t.values = t.values.swapaxes(k, p + f)
        if self.p2_mapping is not None:
            for k, f in enumerate(self.p2_mapping):
                if self.p1_mapping is not None and f in self.p1_mapping:
                    idx = np.zeros(self.values.ndim, dtype=np.bool)
                    idx[self.p1_mapping.index(f)] = True
                    idx[self.p1 + k] = True
                    idx = [slice(None) if x else None for x in idx]
                    t.values = t.values * np.eye(t.values.shape[k])[idx]
                else:
                    t.values = t.values.swapaxes(self.p1 + k, p + f)
        if dim is True:
            idx = tuple(p + np.arange(self.n + self.dim))
            t.dim = 0
        else:
            idx = tuple(p + np.arange(self.n))
        t.values = t.values.sum(idx).transpose()
        t.p1_mapping = None
        t.p2_mapping = None
        t.n = 0
        return t

    def expand(self, xmap, newsize, dim=False):
        """
        Applied either to N or to Dim, accordingly with the 'dim' flag
        """
        assert max(xmap or [-1]) < newsize

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
            p1_mapping = None
            p2_mapping = None
            if self.p1_mapping is not None:
                p1_mapping = [xmap[k] for k in self.p1_mapping]
            if self.p2_mapping is not None:
                p2_mapping = [xmap[k] for k in self.p2_mapping]
            return Tensor(
                self.values[idx],
                p1=self.p1,
                p2=self.p2,
                dim=self.dim,
                p1_mapping=p1_mapping,
                p2_mapping=p2_mapping)

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

    def dot(self, other):

        # import ipdb; ipdb.set_trace()

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

                p = other.p1 + other.p2
                dim = max(self.dim, other.dim)

                # to do, broadcast on dim
                l_idx = [slice(None)] * (p + other.n) + [None] * self.n + \
                        [slice(None)] * other.dim + [None] * (dim - other.dim)
                r_idx = [None] * p + [Ellipsis] + [None] * (dim - self.dim)

                p1_mapping = None
                p2_mapping = None
                val = self.values.copy()
                if self.p1_mapping:
                    for k, f in enumerate(self.p1_mapping):
                        val = val.swapaxes(k, self.p1 + f)
                val = other.values[l_idx] * val[r_idx]

                if self.p1_mapping:
                    for k, f in enumerate(self.p1_mapping):
                        val = val.swapaxes(p + k, p + self.n + f)

                    if other.p1_mapping:
                        p1_mapping = [self.p1_mapping[f]
                                      for f in other.p1_mapping]
                    if other.p2_mapping:
                        p2_mapping = [self.p1_mapping[f]
                                      for f in other.p2_mapping]
                else:
                    if other.p1_mapping:
                        for k, f in enumerate(other.p1_mapping):
                            val = val.swapaxes(k, p + f)
                    if other.p2_mapping:
                        for k, f in enumerate(other.p2_mapping):
                            if other.p1_mapping and f in other.p1_mapping:
                                # overlap
                                idx = np.zeros(val.ndim, dtype=np.bool)
                                idx[other.p1_mapping.index(f)] = True
                                idx[other.p1 + k] = True
                                idx = [slice(None) if x else None for x in idx]
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

        elif other.p1 > 0:

            # P1 -  N E (other)
            # N  P2 M E (self)
            #
            # product
            # P1 N -  - E
            # -  N P2 M E
            #    x

            assert self.p1 == other.n

            p = self.p1 + self.p2

            l_idx = [slice(None)] * (other.p1 + other.n) + \
                    [None] * (self.p2 + self.n) + [slice(None)] * other.dim + \
                    [None] * (dim - other.dim)
            r_idx = [None] * other.p1 + [Ellipsis] + [None] * (dim - self.dim)

            p1_mapping = None
            val = self.values.copy()

            if self.p1_mapping:
                for k, f in enumerate(self.p1_mapping):
                    val = val.swapaxes(k, p + f)
            val = other.values[l_idx] * val[r_idx]

            if self.p1_mapping:
                for k, f in enumerate(self.p1_mapping):
                    val = val.swapaxes(other.p1 + k, other.p1 + p + f)
                if other.p1_mapping:
                    p1_mapping = [self.p1_mapping[f]
                                  for f in other.p1_mapping]
            else:
                if other.p1_mapping:
                    for k, f in enumerate(other.p1_mapping):
                        val = val.swapaxes(k, other.p1 + f)

            val = val.sum(tuple(other.p1 + np.arange(self.n)))
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

        if isinstance(other, (int, float, np.ndarray)):
            return self._bin_op(Tensor(other), op_type)

        scalar_op = False
        if other.values.shape == ():
            scalar_op = True
        elif self.values.shape == ():
            scalar_op = True

        if scalar_op:
            if other.values.shape == ():
                t = self.copy()
            else:
                t = other.copy()
            if op_type == "sum":
                t.values = np.asarray(self.values + other.values)
            elif op_type == "sub":
                t.values = np.asarray(self.values - other.values)
            elif op_type == "mul":
                t.values = np.asarray(self.values * other.values)
            elif op_type == "div":
                t.values = np.asarray(self.values / other.values)
            else:
                raise NotImplementedError
            return t

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
                if op_type == "sum":
                    return GenericTensor(
                        p1, p2, n, dim, [self.copy(), other.copy()])
                elif op_type == "sub":
                    return GenericTensor(
                        p1, p2, n, dim, [self.copy(), -other.copy()])
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
                if op_type == "sum":
                    return GenericTensor(
                        p1, p2, n, dim, [self.copy(), other.copy()])
                elif op_type == "sub":
                    return GenericTensor(
                        p1, p2, n, dim, [self.copy(), -other.copy()])
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

        if op_type == "sum":
            values = self.values[l_idx] + other.values[r_idx]
        elif op_type == "sub":
            values = self.values[l_idx] - other.values[r_idx]
        elif op_type == "mul":
            values = self.values[l_idx] * other.values[r_idx]
        elif op_type == "div":
            values = self.values[l_idx] / other.values[r_idx]
        else:
            raise NotImplementedError

        return Tensor(values, p1=p1, p2=p2, dim=dim,
                      p1_mapping=p1_mapping, p2_mapping=p2_mapping)

    def shape(self):
        return "(p1:%d, p2:%d, n:%d, v:%d)" % \
            (self.p1, self.p2, self.n, self.dim)

    def __getitem__(self, i):
        return self.values[i]

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

    def __add__(self, other):
        return self._bin_op(other, "sum")

    def __sub__(self, other):
        return self._bin_op(other, "sub")

    def __mul__(self, other):
        return self._bin_op(other, "mul")

    def __div__(self, other):
        return self._bin_op(other, "div")


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
