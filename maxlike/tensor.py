import numpy as np

class Tensor:
    def __init__(self, values=0, p1=0, p2=0, v=0,
                 p1_mapping=None, p2_mapping=None):
        self.values = np.asarray(values)
        self.n = self.values.ndim - p1 - p2 - v
        assert self.n >= 0
        self.p1 = p1
        self.p2 = p2
        self.p1_mapping = None
        self.p2_mapping = None
        self.v = v
        if p1_mapping is not None:
            if p1_mapping is True:
                assert p1 == self.n
                self.p1_mapping = range(self.n)
            elif isinstance(p1_mapping, (list, tuple, range)):
                assert len(p1_mapping) == p1
                self.p1_mapping = p1_mapping
            else:
                raise ValueError
        if p2_mapping is not None:
            if p2_mapping is True:
                assert p2 == self.n
                self.p2_mapping = range(self.n)
            elif isinstance(p2_mapping, (list, tuple, range)):
                assert len(p2_mapping) == p2
                self.p2_mapping = p2_mapping
            else:
                raise ValueError

    def sum(self):
        t = self.copy()
        p = self.p1 + self.p2
        if self.p1_mapping is not None:
            for k, f in enumerate(self.p1_mapping):
                t.values = t.values.swapaxes(k, p + f)
        if self.p2_mapping is not None:
            for k, f in enumerate(self.p2_mapping):
                if self.p1_mapping is not None and f in self.p1_mapping:
                    idx = np.zeros(self.values.ndim, dtype=np.bool)
                    idx[self.p1_mapping.index(f)] = True  # isnt it k?
                    idx[self.p1 + k] = True
                    idx = [slice(None) if x else None for x in idx]
                    t.values = t.values * np.eye(t.values.shape[k])[idx]
                else:
                    t.values = t.values.swapaxes(self.p1 + k, p + f)
        idx = tuple(p + np.arange(self.n + self.v))
        t.values = t.values.sum(idx).transpose()
        t.p1_mapping = None
        t.p2_mapping = None
        t.n = 0
        t.v = 0
        return t

    def expand(self, feat_map, ndim):
        assert len(feat_map) == self.n
        assert max(feat_map) < ndim
        idx = [slice(None)] * (self.p1 + self.p2)
        idx += [slice(None) if k in feat_map else None for k in range(ndim)]
        idx += [slice(None)] * self.v
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
            v=self.v,
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
        if (other.p1 > 0) & (other.p2 > 0):
            if self.p1 > 0:
                assert self.p2 == 0
                assert self.p1 == other.n
                # new: p1=other.p1, p2=other.p2, n=self.n
                p = other.p1 + other.p2
                idx_lhs = [Ellipsis] + [None] * self.n
                idx_rhs = [None] * p + [Ellipsis]
                val = (other.values[idx_lhs] * self.values[idx_rhs]).sum(
                    tuple(p + np.arange(other.n)))
                return Tensor(val, p1=other.p1, p2=other.p2)

            elif self.p2 > 0:
                assert self.p1 == 0
                assert self.p2 == other.n
                # new: p1=other.p2, p2=other.p1, n=self.n
                return self.dot(other.transpose()).transpose()
            else:
                raise ValueError
        elif other.p1 > 0:
            assert self.p1 == other.n
            # new: p1=other.p1, p2=other.p2, n=self.n
            idx_lhs = [Ellipsis] + [None] * (self.p2 + self.n)
            idx_rhs = [None] * other.p1 + [Ellipsis]
            val = (other.values[idx_lhs] * self.values[idx_rhs]).sum(
                tuple(other.p1 + np.arange(other.n)))
        elif other.p2 > 0:
            assert self.p2 == other.n
            # new: p1=other.p2, p2=other.p1, n=self.n
            return self.dot(other.transpose()).transpose()
        else:
            raise ValueError

    def copy(self):
        return Tensor(self.values, p1=self.p1, p2=self.p2, v=self.v,
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

        # set new_p1
        p1 = max(self.p1, other.p1)
        if self.p1 == other.p1:
            if self.p1_mapping == other.p1_mapping:
                p1_mapping = self.p1_mapping
            else:
                raise NotImplementedError
        elif other.p1 == 0:
            p1_mapping = self.p1_mapping
        elif self.p1 == 0:
            p1_mapping = other.p1_mapping
        else:
            raise ValueError

        # set new_p2
        p2 = max(self.p2, other.p2)
        if self.p2 == other.p2:
            if self.p2_mapping == other.p2_mapping:
                p2_mapping = self.p2_mapping
            else:
                raise NotImplementedError
        elif other.p2 == 0:
            p2_mapping = self.p2_mapping
        elif self.p2 == 0:
            p2_mapping = other.p2_mapping
        else:
            raise ValueError

        # set new_v
        assert (self.v == 0) | (other.v == 0) | (self.v == other.v)
        v = max(self.v, other.v)

        # adjust the values
        l_idx = self.__reshape_idx(p1, p2, n, v)
        r_idx = other.__reshape_idx(p1, p2, n, v)

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

        return Tensor(values, p1=p1, p2=p2, v=v,
                      p1_mapping=p1_mapping, p2_mapping=p2_mapping)

    def shape(self):
        return "(p1:%d, p2:%d, n:%d, v:%d)" % \
            (self.p1, self.p2, self.n, self.v)

    def __getitem__(self, i):
        return self.values[i]

    def __str__(self):
        s = str(self.values)
        s += "\nshape: "
        s += str((self.p1, self.p2, self.n, self.v))
        return s

    def __repr__(self):  # just displays the shape
        return repr(self.values)

    def __array__(self):
        return self.values

    def __reshape_idx(self, p1, p2, n, v):
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
        if self.v > 0:
            assert self.v == v
            idx += [slice(None)] * v
        elif (self.v == 0) & (v > 0):
            idx += [None] * v
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
        return self._bin_op(-1, "mul")

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


def grad_tensor(values, params, i=0, p1_mapping=None, v=0):
    p1 = params[i].ndim
    if p1_mapping is None:
        idx = [Ellipsis]
    else:
        idx = [None] * p1 + [Ellipsis]
    return Tensor(values[idx], p1=p1, v=v, p1_mapping=p1_mapping)


def hess_tensor(values, params, i=0, j=0,
                p1_mapping=None, p2_mapping=None, v=0):
    p1 = params[i].ndim
    p2 = params[j].ndim
    idx = [slice(None) if p1_mapping is None else None] * p1
    idx += [slice(None) if p2_mapping is None else None] * p2
    idx += [Ellipsis]
    return Tensor(values[idx], p1=p1, p2=p2, v=v,
                  p1_mapping=p1_mapping, p2_mapping=p2_mapping)
