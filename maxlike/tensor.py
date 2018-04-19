import numpy as np


class GenericTensor:
    def __init__(self, p1=0, p2=0, n=0, e=0, elements=None):
        self.p1 = p1
        self.p2 = p2
        self.n = n
        self.e = e

        # all of the elements need to have the same shape
        if elements:
            self.elements = elements
        else:
            self.elements = []

    def sum(self, e=False):
        return Tensor(
            sum([el.sum(e).values for el in self.elements]),
            p1=self.p1, p2=self.p2, e=self.e)

    def expand(self, feat_map, ndim, e=False):
        if e:
            self.e = ndim
        else:
            self.n = ndim

        self.elements = [el.expand(feat_map, ndim, e)
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
            self.p1, self.p2, self.n, self.e,
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
        e = max(self.e, other.e)

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

            return GenericTensor(p1, p2, n, e, new_elements)

        # GenericTensor
        if isinstance(other, GenericTensor):
            gt = self.copy()
            for el in other.elements:
                gt = gt.__add(el, sgn)
            return gt

        raise ValueError

    def shape(self):
        return "(p1:%d, p2:%d, n:%d, v:%d)" % \
            (self.p1, self.p2, self.n, self.e)

    def __add__(self, other):
        return self._add(other, 1.0)

    def __sub__(self, other):
        return self._add(other, -1.0)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return self - other

    def __len__(self):
        return len(self.elements)

    def _scalar_mul(self, a):
        assert isinstance(a, (int, float))
        return GenericTensor(
                self.p1, self.p2, self.n, self.e,
                [el * a for el in self.elements])

    # multiplication / division just with scalars
    def __mul__(self, other):
        return self._scalar_mul(other)

    def __div__(self, other):
        return self._scalar_mul(1.0 / other)

    def __rmul__(self, other):
        return self * other

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __rtruediv__(self, other):
        return self / other



class Tensor:
    def __init__(self, values=0, p1=0, p2=0, e=0,
                 p1_mapping=None, p2_mapping=None):
        self.values = np.asarray(values)
        self.n = self.values.ndim - p1 - p2 - e
        assert self.n >= 0
        self.p1 = p1
        self.p2 = p2
        self.p1_mapping = None
        self.p2_mapping = None
        self.e = e
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

    def sum(self, e=False):
        t = self.copy()
        p = self.p1 + self.p2
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
        if e is True:
            idx = tuple(p + np.arange(self.n + self.e))
            t.e = 0
        else:
            idx = tuple(p + np.arange(self.n))
        t.values = t.values.sum(idx).transpose()
        t.p1_mapping = None
        t.p2_mapping = None
        t.n = 0
        return t

    def expand(self, feat_map, ndim, e=False):
        assert max(feat_map) < ndim
        if e is True:
            assert len(feat_map) == self.e
            idx = [slice(None)] * (self.p1 + self.p2 + self.n)
            idx += [slice(None) if k in feat_map else None for k in range(ndim)]
            return Tensor(
                self.values[idx],
                p1=self.p1,
                p2=self.p2,
                e=ndim,
                p1_mapping=self.p1_mapping,
                p2_mapping=self.p2_mapping)

        assert len(feat_map) == self.n
        idx = [slice(None)] * (self.p1 + self.p2)
        idx += [slice(None) if k in feat_map else None for k in range(ndim)]
        idx += [slice(None)] * self.e
        p1_mapping = None
        p2_mapping = None
        if self.p1_mapping is not None:
            p1_mapping = [feat_map[k] for k in self.p1_mapping]
        if self.p2_mapping is not None:
            p2_mapping = [feat_map[k] for k in self.p2_mapping]
        return Tensor(
            self.values[idx],
            p1=self.p1,
            p2=self.p2,
            e=self.e,
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

        assert (self.e == 0) | (other.e == 0) | (self.e == other.e)
        e = max(self.e, other.e)

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
                e = max(self.e, other.e)

                # to do, broadcast on e
                l_idx = [slice(None)] * (p + other.n) + [None] * self.n + \
                        [slice(None)] * other.e + [None] * (e - other.e)
                r_idx = [None] * p + [Ellipsis] + [None] * (e - self.e)

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
                        p2_mapping = [self.p2_mapping[f]
                                      for f in other.p2_mapping]
                else:
                    if other.p1_mapping:
                        for k, f in enumerate(other.p1_mapping):
                            val = val.swapaxes(k, p + f)
                    if other.p2_mapping:
                        for k, f in enumerate(other.p2_mapping):
                            if other.p1_mapping and f in other.p1_mapping:  # overlap
                                idx = np.zeros(val.ndim, dtype=np.bool)
                                idx[other.p1_mapping.index(f)] = True
                                idx[other.p1 + k] = True
                                idx = [slice(None) if x else None for x in idx]
                                val = val * np.eye(val.shape[k])[idx]
                            else:
                                val = val.swapaxes(other.p1 + k, p + f)  # no overlap

                val = val.sum(tuple(p + np.arange(other.n)))
                return Tensor(val, p1=other.p1, p2=other.p2, e=e,
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
                    [None] * (self.p2 + self.n) + [slice(None)] * other.e + \
                    [None] * (e - other.e)
            r_idx = [None] * other.p1 + [Ellipsis] + [None] * (e - self.e)

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
            return Tensor(val, p1=other.p1, p2=self.p2,
                          p1_mapping=p1_mapping,
                          p2_mapping=self.p2_mapping)

        elif other.p2 > 0:
            return self.transpose().dot(other.transpose()).transpose()

        else:
            raise ValueError

    def copy(self):
        return Tensor(self.values, p1=self.p1, p2=self.p2, e=self.e,
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
        assert (self.e == 0) | (other.e == 0) | (self.e == other.e)
        e = max(self.e, other.e)

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
                    return GenericTensor(p1, p2, n, e, [self.copy(), other.copy()])
                elif op_type == "sub":
                    return GenericTensor(p1, p2, n, e, [self.copy(), -other.copy()])
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
                    return GenericTensor(p1, p2, n, e, [self.copy(), other.copy()])
                elif op_type == "sub":
                    return GenericTensor(p1, p2, n, e, [self.copy(), -other.copy()])
                else:
                    raise NotImplementedError

        elif other.p2 == 0:
            p2_mapping = self.p2_mapping
        elif self.p2 == 0:
            p2_mapping = other.p2_mapping
        else:
            raise ValueError

        # adjust the values
        l_idx = self.__reshape_idx(p1, p2, n, e)
        r_idx = other.__reshape_idx(p1, p2, n, e)

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

        return Tensor(values, p1=p1, p2=p2, e=e,
                      p1_mapping=p1_mapping, p2_mapping=p2_mapping)

    def shape(self):
        return "(p1:%d, p2:%d, n:%d, v:%d)" % \
            (self.p1, self.p2, self.n, self.e)

    def __getitem__(self, i):
        return self.values[i]

    def __str__(self):
        s = str(self.values)
        s += "\nshape: "
        s += str((self.p1, self.p2, self.n, self.e))
        return s

    def __repr__(self):  # just displays the shape
        return repr(self.values)

    def __array__(self):
        return self.values

    def __reshape_idx(self, p1, p2, n, e):
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
        if self.e > 0:
            assert self.e == e
            idx += [slice(None)] * e
        elif (self.e == 0) & (e > 0):
            idx += [None] * e
        return idx

    def __add__(self, other):
        return self._bin_op(other, "sum")

    def __sub__(self, other):
        return self._bin_op(other, "sub")

    def __mul__(self, other):
        return self._bin_op(other, "mul")

    def __div__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        return self._bin_op(other, "div")

    def __neg__(self):
        t = self.copy()
        t.values *= -1
        return t

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return other._bin_op(self, "sub")

    def __rmul__(self, other):
        return self * other

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return other._bin_op(self, "div")


def grad_tensor(values, params, i=0, p1_mapping=None, e=0):
    p1 = params[i].ndim
    if p1_mapping is None:
        idx = [Ellipsis]
    else:
        idx = [None] * p1 + [Ellipsis]
    return Tensor(values[idx], p1=p1, e=e, p1_mapping=p1_mapping)


def hess_tensor(values, params, i=0, j=0,
                p1_mapping=None, p2_mapping=None, e=0):
    p1 = params[i].ndim
    p2 = params[j].ndim
    idx = [slice(None) if p1_mapping is None else None] * p1
    idx += [slice(None) if p2_mapping is None else None] * p2
    idx += [Ellipsis]
    return Tensor(values[idx], p1=p1, p2=p2, e=e,
                  p1_mapping=p1_mapping, p2_mapping=p2_mapping)

if __name__ == "__main__":
    t1 = Tensor(np.ones((1, 1, 4, 5, 6, 7, 2)), p1=2, p2=2, e=1, p1_mapping=True)
    t2 = Tensor(np.ones((6, 7, 2, 2)), p1=2, p2=0)
    t3 = t1.copy()
    t = t1 + t2
    print((t2.dot(t1*5)).sum())  # porque que nao altera... ?

