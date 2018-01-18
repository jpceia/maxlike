import numpy as np
from scipy.misc import factorial
from .tensor import *
from .common import *


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
