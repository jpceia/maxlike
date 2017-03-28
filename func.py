import numpy as np


def eval_func(f):
    def wrapper(obj, params=None, expand=None):
        if params is None:
            params = []
        return f(obj, params)
    return wrapper


def vector_func(g):
    def wrapper(obj, params=None, i=None, expand=None):
        if params is None:
            params = []
        if i is not None:
            return g(obj, params, i)
        else:
            return map(lambda k: g(obj, params, k), range(len(params)))
    return wrapper


def matrix_func(h):
    def wrapper(obj, params=None, i=None, j=None, expand=None):
        if params is None:
            params = []
        if i is not None and j is not None:
            return h(obj, params, i, j)
        else:
            return map(lambda k: map(lambda l: h(obj, params, k, l),
                       range(k + 1)), range(len(params)))
    return wrapper


class ParamMap(list):
    def __init__(self, indexes):
        if isinstance(indexes, int):
            indexes = [indexes]
        assert min(indexes) >= 0
        self.extend(indexes)

    def __call__(self, params):
        return map(params.__getitem__, self)


class Func:
    def eval(self, params):
        raise NotImplementedError

    def grad(self, params, i):
        raise NotImplementedError

    def hess(self, params, i, j):
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

    @eval_func
    def eval(self, params):
        return self.a * self.base.eval(params) + self.b

    @vector_func
    def grad(self, params, i):
        return self.a * self.base.grad(params, i)

    @matrix_func
    def hess(self, params, i=None, j=None):
        return self.a * self.base.hess(params, i, j)


class FuncSum(Func):
    class Atom(Func):
        def __init__(self, ndim, param_map, feat_map, foo):
            if isinstance(feat_map, int):
                feat_map = [feat_map]
            assert isinstance(foo, Func)
            assert min(feat_map) >= 0
            assert max(feat_map) <= ndim
            self.ndim = ndim
            self.param_map = ParamMap(param_map)
            self.feat_map = feat_map
            self.foo = foo
            self.__slice = map(
                lambda x: slice(None) if x else None,
                (np.arange(self.ndim)[:, None] ==
                 np.array(self.feat_map)[None, :]).any(1))

        def eval(self, params):
            return self.foo.eval(self.param_map(params))[self.__slice]

        def grad(self, params, i):
            if i in self.param_map:
                return self.foo.grad(
                    self.param_map(params),
                    self.param_map.index(i))[
                    self.__slice + [Ellipsis]]
            else:
                return np.zeros(params[i].shape)[
                    [None] * self.ndim + [Ellipsis]]

        def hess(self, params, i, j):
            if i in self.param_map and j in self.param_map:
                return self.foo.hess(
                    self.param_map(params),
                    self.param_map.index(i),
                    self.param_map.index(j))[
                    self.__slice + [Ellipsis]]
            else:
                return np.zeros(params[i].shape + params[j].shape)[
                    [None] * self.ndim + [Ellipsis]]

    def __init__(self, ndim):
        self.atoms = []
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
                    map(feat_map.__getitem__, atom.feat_map),
                    atom.foo,
                    w * weight)
        else:
            self.atoms.append((
                weight, self.Atom(self.ndim, param_map, feat_map, foo)))

    @eval_func
    def eval(self, params):
        return sum([w * atom.eval(params) for w, atom in self.atoms]) + self.b

    @vector_func
    def grad(self, params, i):
        return sum([w * atom.grad(params, i) for w, atom in self.atoms])

    @matrix_func
    def hess(self, params, i, j):
        return sum([w * atom.hess(params, i, j) for w, atom in self.atoms])


class Linear(Func):
    def __init__(self):
        self.weight = []

    @eval_func
    def eval(self, params):
        if not isinstance(params, (tuple, list)):
            return sum(params * self.weight[0])
        else:
            return sum([sum(params[i] * self.weight[i])
                        for i in range(len(params))])

    @vector_func
    def grad(self, params, i):
        return self.weight[i]

    @matrix_func
    def hess(self, params, i, j):
        return np.multiply.outer(
            np.zeros(self.weight[j].shape),
            np.zeros(self.weight[i].shape))

    def add_feature(self, shape, weight):
        self.weight.append(weight * np.ones(shape))


class OneHot(Func):
    @eval_func
    def eval(self, params):
        return np.array(params)[0]

    @vector_func
    def grad(self, params, i):
        return np.diag(np.ones(params[0].size)).reshape(params[0].shape * 2)

    @matrix_func
    def hess(self, params, i, j):
        return np.zeros(params[0].shape * 3)


class Constant(Func):
    def __init__(self, vector):
        self.vector = np.array(vector)

    @eval_func
    def eval(self, params):
        return self.vector

    @vector_func
    def grad(self, params, i):
        return np.zeros(())

    @matrix_func
    def hess(self, params, i, j):
        return np.zeros(())


class Vector(Func):
    def __init__(self, vector):
        self.vector = np.array(vector)

    @eval_func
    def eval(self, params):
        return params * self.vector

    @vector_func
    def grad(self, params, i):
        return self.vector

    @matrix_func
    def hess(self, params, i, j):
        return np.zeros(self.vector.size)
