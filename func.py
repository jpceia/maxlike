import numpy as np


def vector_func(g):
    def wrapper(obj, param=None, i=None):
        if param is None:
            param = []
        elif i is not None:
            return g(obj, param, i)
        else:
            return map(lambda k: g(obj, param, k), range(len(param)))
    return wrapper


def matrix_func(h):
    def wrapper(obj, param=None, i=None, j=None):
        if param is None:
            param = []
        if i is not None and j is not None:
            return h(obj, param, i, j)
        else:
            return map(lambda k:
                       map(lambda l: h(obj, param, k, l), range(k + 1)),
                       range(len(param)))
    return wrapper


class Func:
    def eval(self, param):
        raise NotImplementedError

    def grad(self, param, i):
        raise NotImplementedError

    def hess(self, param, i, j):
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

    def eval(self, param):
        return self.a * self.base.eval(param) + self.b

    @vector_func
    def grad(self, param, i):
        return self.a * self.base.grad(param, i)

    @matrix_func
    def hess(self, param, i=None, j=None):
        return self.a * self.base.hess(param, i, j)


class FuncSum(Func):
    class Atom(Func):
        def __init__(self, ndim, param_map, feat_map, foo):
            if isinstance(param_map, int):
                self.param_map = [param_map]
            if isinstance(feat_map, int):
                self.feat_map = [feat_map]
            assert isinstance(foo, Func)
            assert min(feat_map) >= 0
            assert max(feat_map) <= ndim
            self.ndim = ndim
            self.foo = foo
            self.__slice = map(
                lambda x: slice(None) if x else None,
                (np.arange(self.ndim)[:, None] ==
                 np.array(self.feat_map)[None, :]).any(1))

        def eval(self, param):
            return self.foo.eval(map(param.__getitem__,
                                     self.param_map))[self.__slice]

        @vector_func
        def grad(self, param, i):
            if i in self.param_map:
                filt_param = map(param.__getitem__, self.param_map)
                idx = self.param_map.index(i)
                return self.foo.grad(filt_param, idx)[
                    self.__slice + [Ellipsis]]
            else:
                return np.zeros(param[i].shape)[
                    [None] * self.ndim + [Ellipsis]]

        @matrix_func
        def hess(self, param, i, j):
            if i in self.param_map and j in self.param_map:
                filt_param = map(param.__getitem__, self.param_map)
                idx = self.param_map.index(i)
                jdx = self.param_map.index(j)
                return self.foo.hess(filt_param, idx, jdx)[
                    self.__slice + [Ellipsis]]
            else:
                return np.zeros(param[i].shape + param[j].shape)[
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
                    map(param_map.__getitem__, atom.param_map),
                    map(feat_map.__getitem__, atom.feat_map),
                    atom.foo,
                    w * weight)
        else:
            self.atoms.append((
                weight, self.Atom(self.ndim, param_map, feat_map, foo)))

    def eval(self, param):
        return sum([w * atom.eval(param) for w, atom in self.atoms]) + self.b

    def grad(self, param, i):
        return sum([w * atom.grad(param, i) for w, atom in self.atoms])

    def hess(self, param, i, j):
        return sum([w * atom.hess(param, i, j) for w, atom in self.atoms])


class Linear(Func):
    def __init__(self):
        self.weight = []

    def eval(self, param):
        if not isinstance(param, (tuple, list)):
            return sum(param * self.weight[0])
        else:
            return sum([sum(param[i] * self.weight[i])
                        for i in range(len(param))])

    @vector_func
    def grad(self, param, i):
        return self.weight[i]

    @matrix_func
    def hess(self, param, i, j):
        return np.multiply.outer(
            np.zeros(self.weight[j].shape),
            np.zeros(self.weight[i].shape))

    def add_feature(self, shape, weight):
        self.weight.append(weight * np.ones(shape))


class OneHot(Func):
    def eval(self, param):
        return np.array(param)[0]

    @vector_func
    def grad(self, param, i):
        return np.diag(np.ones(param[0].size)).reshape(param[0].shape * 2)

    @matrix_func
    def hess(self, param, i, j):
        return np.zeros(param[0].shape * 3)


class Constant(Func):
    def __init__(self, vector):
        self.vector = np.array(vector)

    def eval(self, param=None):
        return self.vector

    @vector_func
    def grad(self, param, i):
        return np.zeros(())

    @matrix_func
    def hess(self, param, i, j):
        return np.zeros(())


class Vector(Func):
    def __init__(self, vector):
        self.vector = np.array(vector)

    def eval(self, param):
        return param * self.vector

    @vector_func
    def grad(self, param, i):
        return self.vector

    @matrix_func
    def hess(self, param, i, j):
        return np.zeros(self.vector.size)
