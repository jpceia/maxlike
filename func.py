import numpy as np
from collections import defaultdict
from common import *


class Func:
    def __init__(self):
        self.param2feat = defaultdict(list)

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


class Affine(Func):
    def __init__(self, base, a, b):
        if isinstance(base, Affine):
            self.base = base.base
            self.a = a * base.a
            self.b = a * base.b + b
        else:
            self.base = base
            self.a = a
            self.b = b

    @property
    def param2feat(self):
        return self.base.param2feat

    @call_func
    def __call__(self, params, **kwargs):
        return self.a * self.base(params) + self.b

    @vector_func
    def grad(self, params, i, **kwargs):
        return self.a * self.base.grad(params, i, **kwargs)

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return self.a * self.base.hess(params, i, j, **kwargs)


class FuncSum(Func):
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
            self.__slice = map(
                lambda x: slice(None) if x else None,
                (np.arange(self.ndim)[:, None] ==
                 np.array(self.feat_map)[None, :]).any(1))

        def __call__(self, params, **kwargs):
            return self.foo(self.param_map(params))[self.__slice]

        @property
        def param2feat(self):
            return {self.param_map(p)[0]: self.feat_map(f)
                    for p, f in self.foo.param2feat.iteritems()}

        def grad(self, params, i, **kwargs):
            if i in self.param_map:
                return self.foo.grad(
                    self.param_map(params),
                    self.param_map.index(i),
                    **kwargs)[[Ellipsis] + self.__slice]
            else:
                return np.zeros(())

        def hess(self, params, i, j, **kwargs):
            if i in self.param_map and j in self.param_map:
                return self.foo.hess(
                    self.param_map(params),
                    self.param_map.index(i),
                    self.param_map.index(j),
                    **kwargs)[[Ellipsis] + self.__slice]
            else:
                return np.zeros(())

    def __init__(self, ndim):
        Func.__init__(self)
        self.atoms = []
        self.param2feat = defaultdict(list)
        self.ndim = ndim
        self.b = 0

    def add(self, param_map, feat_map, foo, weight=1.0):
        """
        Adds a factor to FuncSum object.

        Parameters
        ----------
        param_map: int, list
            index of parameters that 'foo' accepts.
        feat_map:
            index of features that 'foo' returns.
        foo: Func
            function to add to the FuncSum object.
        weight: double
            value to multiply for.
        """
        if isinstance(param_map, int):
            param_map = [param_map]
        if isinstance(feat_map, int):
            feat_map = [feat_map]

        if isinstance(foo, Affine):
            self.b += foo.b
            self.add(param_map, feat_map, foo.base, weight * foo.a)
        elif isinstance(foo, FuncSum):
            self.b += foo.b
            for w, atom in foo.atoms:
                self.add(
                    atom.param_map(param_map),
                    atom.feat_map(feat_map),
                    atom.foo,
                    w * weight)
        else:
            for p, f in foo.param2feat.iteritems():
                if param_map[p] in self.params:
                    raise IndexError
                    # as alternative, just send a warning and
                    # loose the diag property
            for p, f in self.param2feat.iteritems():
                if p in param_map:
                    raise IndexError
                    # as alternative, just send a warning and
                    # loose the diag property
            for p, f in foo.param2feat.iteritems():
                self.param2feat[param_map[p]] = IndexMap(f)(feat_map)
            self.atoms.append((
                weight, self.Atom(self.ndim, param_map, feat_map, foo)))

    @property
    def params(self):
        return set([p for w, atom in self.atoms for p in atom.param_map])

    def vector_sum(self, vector, i):
        param2feat = self.param2feat.get(i, [])
        for k in range(len(param2feat)):
            vector = vector.swapaxes(k, param2feat[k] - self.ndim)
        return vector.sum(tuple(-np.arange(self.ndim) - 1))

    def matrix_sum(self, matrix, i, j):
        param2feat_i = self.param2feat.get(i, [])
        param2feat_j = self.param2feat.get(j, [])
        pdim_i = len(param2feat_i)
        pdim_j = len(param2feat_j)
        for k in range(pdim_i):
            matrix = matrix.swapaxes(
                k - pdim_i - self.ndim,
                param2feat_i[k] - self.ndim)
        for k in range(pdim_j):
            f = param2feat_j[k]
            if f in param2feat_i:
                idx = np.zeros(pdim_i + pdim_j + self.ndim, dtype=np.bool)
                idx[param2feat_j.index(f)] = True
                idx[pdim_j + k] = True
                idx = map(lambda b: slice(None) if b else None, idx)
                matrix = matrix * np.eye(matrix.shape[pdim_j + k])[idx]
            else:
                matrix = matrix.swapaxes(k, f - self.ndim)
        return matrix.sum(tuple(-np.arange(self.ndim) - 1))

    @call_func
    def __call__(self, params, **kwargs):
        return sum([w * atom(params) for w, atom in self.atoms]) + self.b

    @vector_func
    def grad(self, params, i, **kwargs):
        diag_param = self.param2feat.keys()
        return sum([w * atom.grad(params, i, diag=(i in diag_param))
                    for w, atom in self.atoms])

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        diag_param = self.param2feat.keys()
        diag_i = i in diag_param
        diag_j = j in diag_param
        return sum([w * atom.hess(params, i, j, diag_i=diag_i, diag_j=diag_j)
                    for w, atom in self.atoms])


class Linear(Func):
    def __init__(self, weight_list=None):
        Func.__init__(self)
        if weight_list is None:
            weight_list = []
        elif not isinstance(weight_list, (list, tuple)):
            weight_list = [weight_list]
        self.weight = weight_list

    @call_func
    def __call__(self, params, **kwargs):
        return sum([sum(params[i] * self.weight[i])
                    for i in range(len(params))])

    @vector_func
    def grad(self, params, i, **kwargs):
        return self.weight[i] * np.ones(params[i].shape)

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return np.multiply.outer(
            np.zeros(params[j].shape),
            np.zeros(params[i].shape))

    def add_feature(self, weight):
        self.weight.append(weight)


class Quadratic(Func):  # TODO : expand this class to allow more generic stuff
    def __init__(self, weight=None):
        Func.__init__(self)
        self.weight = weight

    @call_func
    def __call__(self, params, **kwargs):
        return ((self.weight * params[0]) * (self.weight * params[0])).sum()

    @vector_func
    def grad(self, params, i, **kwargs):
        return 2 * self.weight * self.weight * params[0]

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return 2 * np.diag(np.ones(params[0].size)) * self.weight * self.weight


class OneHot(Func):
    def __init__(self, diag=True):
        Func.__init__(self)
        if diag:
            self.param2feat[0].append(0)

    @call_func
    def __call__(self, params, **kwargs):
        return np.array(params[0])

    @vector_func
    def grad(self, params, i, diag=False, **kwargs):
        if diag:
            return np.ones(params[0].shape)[
                params[0].ndim * [None] + [Ellipsis]]
        else:
            return np.diag(np.ones(params[0].size)).\
                reshape(2 * params[0].shape)

    @matrix_func
    def hess(self, params, i, j, diag_i=False, diag_j=False, **kwargs):
        if diag_i:
            return np.zeros(params[0].shape)[
                2 * params[0].ndim * [None] + [Ellipsis]]
        else:
            return np.zeros(params[0].shape * 3)


class Constant(Func):
    def __init__(self, vector):
        Func.__init__(self)
        self.vector = np.array(vector)

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
        Func.__init__(self)
        self.vector = np.array(vector)

    @call_func
    def __call__(self, params, **kwargs):
        return params * self.vector

    @vector_func
    def grad(self, params, i, **kwargs):
        return self.vector

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        return np.zeros(self.vector.shape)
