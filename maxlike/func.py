import numpy as np
from scipy.misc import factorial
from .common import *


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
        t.values = t.values.sum(tuple(p + np.arange(self.n + self.v)))
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
            p1_mapping = IndexMap(self.p1_mapping)(feat_map)
        if self.p2_mapping is not None:
            p2_mapping = IndexMap(self.p2_mapping)(feat_map)
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
                t.values = np.moveaxis(t.values, 0, self.p1 + k)
        t.p1 = self.p2
        t.p2 = self.p1
        t.p1_mapping = self.p2_mapping
        t.p2_mapping = self.p1_mapping
        return t

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


class Func(object):
    def __call__(self, params, **kwargs):
        raise NotImplementedError

    def grad(self, params, i, **kwargs):
        raise NotImplementedError

    def hess(self, params, i, j, **kwargs):
        raise NotImplementedError

    def __add__(self, b):
        return Affine(self, 1, b)

    def __sub__(self, b):
        return Affine(self, 1, -b)

    def __neg__(self):
        return Affine(self, -1, 0)

    def __mul__(self, a):
        return Affine(self, a, 0)

    def __div__(self, a):
        return Affine(self, 1.0 / a, 0)


# maybe this can be replaced by a metaclass
class Atom(Func):
    def __init__(self, ndim, param_map, feat_map, foo):
        assert isinstance(foo, Func)
        # assert max(feat_map) <= ndim
        self.ndim = ndim
        self.param_map = IndexMap(param_map)
        self.feat_map = feat_map
        self.foo = foo

    def __call__(self, params, **kwargs):
        return self.foo(self.param_map(params)).\
            expand(self.feat_map, self.ndim)

    def grad(self, params, i, **kwargs):
        try:
            idx = self.param_map.index(i)
        except ValueError:
            return Tensor()
        else:
            return self.foo.grad(
                self.param_map(params), idx,
                **kwargs).expand(self.feat_map, self.ndim)

    def hess(self, params, i, j, **kwargs):
        try:
            idx = self.param_map.index(i)
            jdx = self.param_map.index(j)
        except ValueError:
            return Tensor()
        else:
            return self.foo.hess(
                self.param_map(params), idx, jdx,
                **kwargs).expand(self.feat_map, self.ndim)


class Affine(Func):
    def __init__(self, base, a=1, b=0):
        if isinstance(base, Affine):
            self.base = base.base
            self.a = a * base.a
            self.b = a * base.b + b
        else:
            self.base = base
            self.a = a
            self.b = b

    @property
    def param_feat(self):
        return self.base.param_feat

    @call_func
    def __call__(self, params, **kwargs):
        return self.a * self.base(params) + self.b

    @vector_func
    def grad(self, params, i, **kwargs):
        return self.a * self.base.grad(params, i, **kwargs)

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return self.a * self.base.hess(params, i, j, **kwargs)


class Sum(Func):
    def __init__(self, ndim):
        self.atoms = []
        self.ndim = ndim
        self.b = 0

    def add(self, param_map, feat_map, foo, weight=1.0):
        """
        Adds a factor to Ensemble object.

        Parameters
        ----------
        param_map: int, list
            index of parameters that 'foo' accepts.
        feat_map:
            index of features that 'foo' returns.
        foo: Func
            function to add to the Ensemble object.
        weight: double
            weight
        """
        if isinstance(param_map, int):
            param_map = [param_map]
        if isinstance(feat_map, int):
            feat_map = [feat_map]

        if isinstance(foo, Affine):
            self.b += foo.b
            self.add(param_map, feat_map, foo.base, weight * foo.a)
        elif isinstance(foo, Sum):
            self.b += foo.b
            for w, atom in foo.atoms:
                Ensemble.add(
                    self,
                    atom.param_map(param_map),
                    atom.feat_map(feat_map),
                    atom.foo,
                    w * weight)
        else:
            self.atoms.append((
                weight, Atom(self.ndim, param_map, feat_map, foo)))
        return self

    @call_func
    def __call__(self, params, **kwargs):
        return sum((w * atom(params) for w, atom in self.atoms)) + self.b

    @vector_func
    def grad(self, params, i, **kwargs):
        return sum((w * atom.grad(params, i) for w, atom in self.atoms))

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return sum((w * atom.hess(params, i, j) for w, atom in self.atoms))


class Linear(Func):
    def __init__(self, weight_list=None):
        if weight_list is None:
            weight_list = []
        elif not isinstance(weight_list, (list, tuple)):
            weight_list = [weight_list]
        self.weights = weight_list

    @call_func
    def __call__(self, params, **kwargs):
        return sum(((w * param).sum()
                    for w, param in zip(params, self.weights)))

    @vector_func
    def grad(self, params, i, **kwargs):
        return grad_tensor(self.weights[i] * np.ones(params[i].shape),
                           params, i)

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return hess_tensor(np.multiply.outer(
            np.zeros(params[j].shape),
            np.zeros(params[i].shape)),
            params, i, j)

    def add_feature(self, weight):
        self.weights.append(weight)


class Quadratic(Func):  # TODO : expand this class to allow more generic stuff
    def __init__(self, weight=None):
        self.weight = weight

    @call_func
    def __call__(self, params, **kwargs):
        return sum(((np.dot(self.weight, param) *
                     np.dot(self.weight, param)).sum()
                    for param in params))

    @vector_func
    def grad(self, params, i, **kwargs):
        return 2 * np.dot(self.weight, np.dot(self.weight, params[i]))

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return 2 * np.dot(self.weight, self.weight) * \
            np.ones(params[i].shape + params[j].shape)


class Encode(Func):
    @call_func
    def __call__(self, params, **kwargs):
        return Tensor(params[0])

    @vector_func
    def grad(self, params, i, **kwargs):
        return grad_tensor(np.ones(params[0].shape), params, i, True)

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return hess_tensor(np.zeros(params[0].shape), params, i, j, True, True)


class Constant(Func):
    def __init__(self, vector):
        self.vector = np.asarray(vector)
        self.compact = True

    @call_func
    def __call__(self, params, **kwargs):
        return Tensor(self.vector)

    @vector_func
    def grad(self, params, i, **kwargs):
        return Tensor(np.zeros(self.vector.shape))

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return Tensor(np.zeros(self.vector.shape))


class Vector(Func):
    def __init__(self, vector):
        self.vector = np.asarray(vector)
        self.compact = True

    @call_func
    def __call__(self, params, **kwargs):
        return Tensor(params[0] * self.vector)

    @vector_func
    def grad(self, params, i, **kwargs):
        return grad_tensor(self.vector, params, i, None)

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return hess_tensor(np.zeros(self.vector.shape),
                           params, i, j, None, None)


class Exp(Func):
    def __init__(self, size=10):
        super(Exp, self).__init__()
        self.compact = True

    @call_func
    def __call__(self, params, **kwargs):
        return Tensor(np.exp(np.asarray(params[0])))

    @vector_func
    def grad(self, params, i, **kwargs):
        return grad_tensor(np.exp(np.asarray(params[0])), params, i, True)

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return hess_tensor(np.exp(np.asarray(params[0])),
                           params, i, j, True, True)


class PoissonVector(Func):
    """
    Generates a vector

    """
    def __init__(self, size=10):
        self.size = size
        self.compact = True

    @call_func
    def __call__(self, params, **kwargs):
        a = np.asarray(params[0])
        rng = np.arange(self.size)
        return Tensor(np.exp(-a)[..., None] * (a[..., None] ** rng) /
                      factorial(rng), v=1)

    @vector_func
    def grad(self, params, i, **kwargs):
        vec = self(params).values
        return grad_tensor(np.insert(vec[..., :-1], 0, 0, -1) - vec,
                           params, i, True, v=1)

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        vec = self.grad(params, i).values[0]
        return hess_tensor(np.insert(vec[..., :-1], 0, 0, -1) - vec,
                           params, i, j, True, True, v=1)
