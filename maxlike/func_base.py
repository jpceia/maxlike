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

    def __call__(self, params):
        return self.a * self.base(params) + self.b

    def grad(self, params, i):
        return self.a * self.base.grad(params, i)

    def hess(self, params, i, j):
        return self.a * self.base.hess(params, i, j)


class Compose(Func):

    def __init__(self, f, g_list):
        if not isinstance(g_list, (list, tuple)):
            self.g_list = [g_list]
        else:
            self.g_list = g_list
        self.f = f

    def __f_arg(self, params):
        return [g(params) for g in self.g_list]

    def __call__(self, params):
        return self.f(self.__f_arg(params))

    def grad(self, params, i):
        f_arg = self.__f_arg(params)
        return sum([self.f.grad(f_arg, k).dot(g.grad(params, i).drop_dim())
                    for k, g in enumerate(self.g_list)])

    def hess(self, params, i, j):
        f_arg = self.__f_arg(params)
        h_val = 0
        for k, g_k in enumerate(self.g_list):
            for l, g_l in enumerate(self.g_list):
                h_val += self.f.hess(f_arg, k, l).\
                    dot(g_k.grad(params, i).drop_dim()).\
                    dot(g_l.grad(params, j).drop_dim().transpose())
            h_val += self.f.grad(f_arg, k).\
                    dot(g_k.hess(params, i, j).drop_dim())
        return h_val
