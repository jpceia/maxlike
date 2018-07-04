from six import with_metaclass
from functools import wraps


def call_func(f):
    @wraps(f)
    def wrapper(obj, params=None):
        if params is None:
            params = []
        elif not isinstance(params, (tuple, list)):
            params = [params]
        return f(obj, params)
    return wrapper


def vector_func(g):
    @wraps(g)
    def wrapper(obj, params=None, i=None):
        if params is None:
            params = []
        elif not isinstance(params, (tuple, list)):
            params = [params]
        if i is not None:
            return g(obj, params, i)
        return [g(obj, params, k) for k in range(len(params))]
    return wrapper


def matrix_func(h):
    @wraps(h)
    def wrapper(obj, params=None, i=None, j=None):
        if params is None:
            params = []
        elif not isinstance(params, (tuple, list)):
            params = [params]
        if i is not None and j is not None:
            return h(obj, params, i, j)
        return [[h(obj, params, k, l)
                 for l in range(k + 1)]
                for k in range(len(params))]
    return wrapper


class FuncMeta(type):

    def __new__(cls, name, bases, attrs, **kwargs):
        attrs['__call__'] = call_func(attrs['__call__'])
        if 'grad' in attrs:
            attrs['grad'] = vector_func(attrs['grad'])
        if 'hess' in attrs:
            attrs['hess'] = matrix_func(attrs['hess'])
        return type.__new__(cls, name, bases, attrs, **kwargs)


class Func(with_metaclass(FuncMeta, object)):

    def __call__(self, params):
        raise NotImplementedError

    def grad(self, params, i):
        raise NotImplementedError

    def hess(self, params, i, j):
        raise NotImplementedError

    def __add__(self, b):
        return Affine(self, 1, b)

    def __radd__(self, b):
        return Affine(self, 1, b)

    def __sub__(self, b):
        return Affine(self, 1, -b)

    def __rsub__(self, b):
        return Affine(self, -1, b)

    def __neg__(self):
        return Affine(self, -1, 0)

    def __mul__(self, a):
        return Affine(self, a, 0)

    def __rmul__(self, a):
        return Affine(self, a, 0)

    def __div__(self, a):
        return Affine(self, 1.0 / a, 0)

    def __matmul__(self, other):
        return Compose(self, other)
