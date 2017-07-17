import numpy as np
from scipy.misc import factorial
from collections import defaultdict
from .common import *


class Func(object):
    def __init__(self):
        self.param_feat = defaultdict(list)

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


class Atom(Func):
    def __init__(self, ndim, param_map, feat_map, foo):
        param_map = IndexMap(param_map)
        feat_map = IndexMap(feat_map)
        assert isinstance(foo, Func)
        # assert max(feat_map) <= ndim
        self.ndim = ndim
        self.param_map = param_map
        self.feat_map = feat_map
        self.foo = foo
        self.__slice = [
            slice(None) if x else None
            for x in (np.arange(self.ndim)[:, None] ==
                      np.asarray(self.feat_map)[None, :]).any(1)]

    def __call__(self, params, **kwargs):
        return self.foo(self.param_map(params))[self.__slice]

    @property
    def param_feat(self):
        return {self.param_map(p)[0]: self.feat_map(f)
                for p, f in self.foo.param_feat.items()}

    def grad(self, params, i, **kwargs):
        try:
            idx = self.param_map.index(i)
        except ValueError:
            return np.zeros(())
        else:
            return self.foo.grad(
                self.param_map(params), idx,
                **kwargs)[[Ellipsis] + self.__slice]

    def hess(self, params, i, j, **kwargs):
        try:
            idx = self.param_map.index(i)
            jdx = self.param_map.index(j)
        except ValueError:
            return np.zeros(())
        else:
            return self.foo.hess(
                self.param_map(params), idx, jdx,
                **kwargs)[[Ellipsis] + self.__slice]


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


class Ensemble(Func):
    def __init__(self, ndim):
        self.atoms = []
        self.param_feat = defaultdict(list)
        self.ndim = ndim

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
        for k, p in enumerate(param_map):
            f_new = IndexMap(foo.param_feat.get(k, []))(feat_map)
            if p in set([u for w, atom in self.atoms
                         for u in atom.param_map]):
                f = self.param_feat.get(p, [])
                if f != f_new and p in self.param_feat:
                    del self.param_feat[p]
            elif k in list(foo.param_feat):
                self.param_feat[p] = f_new

        self.atoms.append((
            weight, Atom(self.ndim, param_map, feat_map, foo)))

        return self

    @call_func
    def __call__(self, params, k=0, **kwargs):
        return self.atoms[k][1](params)

    @vector_func
    def grad(self, params, i, k=0, **kwargs):
        diag_param = self.param_feat.keys()
        return self.atoms[k][1].grad(params, i, diag=(i in diag_param))

    @matrix_func
    def hess(self, params, i, j, k=0, **kwargs):
        diag_param = self.param_feat.keys()
        diag_i = i in diag_param
        diag_j = j in diag_param
        return self.atoms[k][1].hess(params, i, j,
                                     diag_i=diag_i, diag_j=diag_j)


class Sum(Ensemble):
    def __init__(self, ndim):
        super(Sum, self).__init__(ndim)
        self.b = 0

    def add(self, param_map, feat_map, foo, weight=1.0):
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
            Ensemble.add(self, param_map, feat_map, foo, weight)
        return self

    @call_func
    def __call__(self, params, **kwargs):
        return sum((w * atom(params) for w, atom in self.atoms)) + self.b

    @vector_func
    def grad(self, params, i, **kwargs):
        diag_param = self.param_feat.keys()
        return sum((w * atom.grad(params, i, diag=(i in diag_param))
                    for w, atom in self.atoms))

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        diag_param = self.param_feat.keys()
        diag_i = i in diag_param
        diag_j = j in diag_param
        return sum((w * atom.hess(params, i, j, diag_i=diag_i, diag_j=diag_j)
                    for w, atom in self.atoms))


class Linear(Func):
    def __init__(self, weight_list=None):
        super(Linear, self).__init__()
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
        return self.weights[i] * np.ones(params[i].shape)

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return np.multiply.outer(
            np.zeros(params[j].shape),
            np.zeros(params[i].shape))

    def add_feature(self, weight):
        self.weights.append(weight)


class Quadratic(Func):  # TODO : expand this class to allow more generic stuff
    def __init__(self, weight=None):
        super(Quadratic, self).__init__()
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
    def __init__(self, diag=True):
        super(Encode, self).__init__()
        if diag:
            self.param_feat[0].append(0)

    @call_func
    def __call__(self, params, **kwargs):
        return np.asarray(params[0])

    @vector_func
    def grad(self, params, i, diag=False, **kwargs):
        if diag:
            return np.ones(params[0].shape)[
                params[0].ndim * [None] + [Ellipsis]]
        return np.diag(np.ones(params[0].size)).reshape(2 * params[0].shape)

    @matrix_func
    def hess(self, params, i, j, diag_i=False, diag_j=False, **kwargs):
        if diag_i:
            return np.zeros(params[0].shape)[
                2 * params[0].ndim * [None] + [Ellipsis]]
        return np.zeros(params[0].shape * 3)


class Constant(Func):
    def __init__(self, vector):
        super(Constant, self).__init__()
        self.vector = np.asarray(vector)

    @call_func
    def __call__(self, params, **kwargs):
        return self.vector

    @vector_func
    def grad(self, params, i, **kwargs):
        return np.zeros(())

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return np.zeros(())


class Vector(Func):
    def __init__(self, vector):
        super(Vector, self).__init__()
        self.vector = np.asarray(vector)

    @call_func
    def __call__(self, params, **kwargs):
        return params * self.vector

    @vector_func
    def grad(self, params, i, **kwargs):
        return self.vector

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return np.zeros(self.vector.shape)


class PoissonVector(Func):
    def __init__(self, size=10):
        super(PoissonVector, self).__init__()
        self.size = size

    @call_func
    def __call__(self, params, **kwargs):
        a = np.asarray(params[0])
        rng = np.arange(self.size)
        return np.exp(-a)[..., None] * (a[..., None] ** rng) / factorial(rng)

    @vector_func
    def grad(self, params, i, **kwargs):
        val = self(params)
        return np.insert(val[..., :-1], 0, 0, -1) - val

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        grad = self.grad(params, i, j)
        return np.insert(grad[..., :-1], 0, 0, -1) - grad


class SkellamMatrix(Func):
    def __init__(self, size=10):
        super(SkellamMatrix, self).__init__()
        self.poi_a = PoissonVector(size)
        self.poi_b = PoissonVector(size)

    @staticmethod
    def __parseparams(params):
        a = np.asarray(params[0])
        b = np.asarray(params[1])
        assert a.shape == b.shape
        return [a], [b]

    @call_func
    def __call__(self, params, **kwargs):
        a, b = self.__parseparams(params)
        return self.poi_a(a)[..., None] * self.poi_b(b)[..., None, :]

    @vector_func
    def grad(self, params, i, **kwargs):
        a, b = self.__parseparams(params)
        if i == 0:
            return self.poi_a.grad(a, 0)[..., None] * self.poi_b(b)[..., None, :]
        elif i == 1:
            return self.poi_a(a)[..., None] * self.poi_b.grad(b, 0)[..., None, :]
        raise IndexError

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        a, b = self.__parseparams(params)
        if (i, j) == (0, 0):
            return self.poi_a.hess(a, 0, 0)[..., None] * self.poi_b(b)[..., None, :]
        elif (i, j) == (1, 0):
            return self.poi_a.grad(a, 0)[..., None] * self.poi_b.grad(b, 0)[..., None, :]
        elif (i, j) == (1, 1):
            return self.poi_a(a)[..., None] * self.poi_b.hess(b, 0, 0)[..., None, :]
        raise IndexError
