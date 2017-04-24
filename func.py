import numpy as np
from collections import defaultdict
from common import *


class Func:
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


class FuncSum(Func):
    class Atom(Func):
        def __init__(self, ndim, param_map, feat_map, foo):
            param_map = IndexMap(param_map)
            feat_map = IndexMap(feat_map)
            assert isinstance(foo, Func)
            assert max(feat_map) <= ndim
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
        def param_feat(self):
            return {self.param_map(p)[0]: self.feat_map(f)
                    for p, f in self.foo.param_feat.iteritems()}

        def grad(self, params, i, **kwargs):
            if i in self.param_map:
                return self.foo.grad(
                    self.param_map(params),
                    self.param_map.index(i),
                    **kwargs)[[Ellipsis] + self.__slice]
            else:
                return np.zeros(params[i].shape)[
                    [Ellipsis] + [None] * self.ndim]

        def hess(self, params, i, j, **kwargs):
            if i in self.param_map and j in self.param_map:
                return self.foo.hess(
                    self.param_map(params),
                    self.param_map.index(i),
                    self.param_map.index(j),
                    **kwargs)[[Ellipsis] + self.__slice]
            else:
                return np.zeros(params[i].shape + params[j].shape)[
                    [Ellipsis] + [None] * self.ndim]

    def __init__(self, ndim):
        Func.__init__(self)
        self.atoms = []
        self.param_feat = defaultdict(list)
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
            for p, f in foo.param_feat.iteritems():
                if param_map.index(p) in self.params:
                    raise IndexError
                    # An exceptional case would be when there is a total match
                    # between two param_feat and IndexMap(param_map),
                    # IndexMap(feat_map)
            for p, f in self.param_feat.iteritems():
                if p in param_map:
                    raise IndexError
                    # An exceptional case would be when there is a total match
                    # between two param_feat and IndexMap(param_map),
                    # IndexMap(feat_map)
            for p, f in foo.param_feat.iteritems():
                self.param_feat[param_map.index(p)] = IndexMap(feat_map)(f)
            self.atoms.append((
                weight, self.Atom(self.ndim, param_map, feat_map, foo)))

    @property
    def params(self):
        return set([p for atom in self.atoms for p in atom.param_map])

    def sum_grad(self, model_grad, i):
        param_feat = self.param_feat.get(i, [])
        for k in range(len(param_feat)):
            model_grad.swap(k, param_feat[k] - self.ndim)
        return model_grad.sum(tuple(-np.arange(self.ndim) - 1))

    def sum_hess(self, model_hess, i, j):
        param_feat_i = self.param_feat.get(i, [])
        param_feat_j = self.param_feat.get(j, [])
        pdim_i = len(param_feat_i)
        pdim_j = len(param_feat_j)
        for k in range(pdim_i):
            model_hess.swap(k, param_feat_i[k] - self.ndim)
        for k in range(pdim_j):
            f = param_feat_j[k]
            if f in param_feat_i:
                idx = np.zeros(pdim_i + pdim_j + self.ndim, dtype=np.bool)
                idx[f] = True
                idx[pdim_i + param_feat_i.index(f)] = True
                idx = map(lambda b: None if b else slice(None), idx)
                model_hess *= np.eye(model_hess.shape[f])[idx]
            else:
                model_hess.swap(k - self.ndim - pdim_j, f - self.ndim)
        return model_hess.sum(tuple(-np.arange(self.ndim) - 1))

    @call_func
    def __call__(self, params, **kwargs):
        return sum([w * atom(params) for w, atom in self.atoms]) + self.b

    @vector_func
    def grad(self, params, i, **kwargs):
        diag_param = self.param_feat.keys()
        return sum([w * atom.grad(params, i, diag=(i in diag_param))
                    for w, atom in self.atoms])

    @matrix_func
    def hess(self, params, i, j, **kwargs):
        diag_param = self.param_feat.keys()
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


class OneHot(Func):
    def __init__(self):
        Func.__init__(self)

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
