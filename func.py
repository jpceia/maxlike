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
    def __init__(self):
        pass

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
