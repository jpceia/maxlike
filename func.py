import numpy as np


class func:
    def __init__(self):
        self.params = []

    def eval(self, param, *args, **kargs):
        raise NotImplementedError

    def grad(self, param, *args, **kargs):
        raise NotImplementedError

    def hess(self, param, *args, **kargs):
        raise NotImplementedError

    @staticmethod
    def _wrap(param):
        if isinstance(param, (tuple, list)):
            return param
        else:
            return [param]


class linear(func):
    def __init__(self):
        self.weight = []
        self.level = 0

    def eval(self, param):
        param = self._wrap(param)
        return sum([sum(self.weight[i] * param[i])
                    for i in range(len(param))]) - self.level

    def grad(self, param):
        return self.weight

    def hess(self, param, i=None, j=None):
        if i is not None and j is not None:
            return np.multiply.outer(
                np.zeros(self.weight[j].shape),
                np.zeros(self.weight[i].shape))
        hess = []
        for i in xrange(len(self.weight)):
            hess_line = []
            for j in xrange(i + 1):
                hess_line.append(np.multiply.outer(
                    np.zeros(self.weight[j].shape),
                    np.zeros(self.weight[i].shape)))
            hess.append(hess_line)
        return hess

    def add_feature(self, shape, weight):
        self.weight.append(weight * np.ones(shape))

    def set_level(self, c):
        self.level = c


class quadratic(func):
    def __init__(self):
        self.weight = []
        self.level = 0

    def eval(self, param):
        pass

    def grad(self, param):
        pass

    def hess(self, param):
        pass

    def add_feature(self, shape, weight):
        pass


class onehot(func):
    def __init__(self):
        pass

    def eval(self, param):
        return np.array(param)[0]

    def grad(self, param):
        return np.diag(
            np.ones(param[0].size)).reshape(param[0].shape * 2)

    def hess(self, param, i=None, j=None):
        H = np.zeros(param[0].shape * 3)
        if i is not None and j is not None:
            return H
        else:
            return [[H]]


class constant(func):
    def __init__(self, vector):
        self.vector = np.array(vector)

    def eval(self, param=None):
        return self.vector

    def grad(self, param=None):
        return np.zeros(())

    def hess(self, param=None, i=None, j=None):
        H = np.zeros(())
        if i is not None and j is not None:
            return H
        else:
            return [[H]]


class vector(func):
    def __init__(self, vector):
        self.vector = np.array(vector)

    def eval(self, param):
        return param * self.vector

    def grad(self, param):
        return self.vector

    def hess(self, param, i=None, j=None):
        H = np.zeros((self.vector.size))
        if i is not None and j is not None:
            return H
        else:
            return [[H]]
3
