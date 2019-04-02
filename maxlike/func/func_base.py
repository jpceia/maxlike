import numpy as np
import inspect
from six import with_metaclass
from types import MethodType
from array import array
from functools import wraps
from ..tensor import Tensor


def null_func(obj, *args, **kwargs):
    return Tensor(0)

def null_method(obj):
    return MethodType(null_func, obj)

def isnull(foo):
    return getattr(foo, 'null', False)

null_func.null = True  # dummy attribute


def grad_tensor(values, params, i=0, p1_mapping=None, dim=0):

    p1 = np.asarray(params[i]).ndim
    if p1_mapping is None:
        idx = [...]
        p1_mapping = array('b')
    else:
        idx = [None] * p1 + [...]
        if p1_mapping is True:
            p1_mapping = array('b', range(p1))
        elif isinstance(p1_mapping, (list, tuple)):
            p1_mapping = array('b', p1_mapping)
        else:
            raise ValueError("Invalid mapping")
        
    return Tensor(values[tuple(idx)], p1=p1, dim=dim, p1_mapping=p1_mapping)


def hess_tensor(values, params, i=0, j=0,
                p1_mapping=None, p2_mapping=None, dim=0):
    p1 = np.asarray(params[i]).ndim
    p2 = np.asarray(params[j]).ndim
    idx  = [slice(None) if p1_mapping is None else None] * p1
    idx += [slice(None) if p2_mapping is None else None] * p2
    idx += [...]

    if p1_mapping is None:
        p1_mapping = array('b')
    else:
        if p1_mapping is True:
            p1_mapping = array('b', range(p1))
        elif isinstance(p1_mapping, (list, tuple)):
            p1_mapping = array('b', p1_mapping)
        else:
            raise ValueError("Invalid mapping")

    if p2_mapping is None:
        p2_mapping = array('b')
    else:
        if p2_mapping is True:
            p2_mapping = array('b', range(p2))
        elif isinstance(p2_mapping, (list, tuple)):
            p2_mapping = array('b', p2_mapping)
        else:
            raise ValueError("Invalid mapping")

    return Tensor(values[tuple(idx)], p1=p1, p2=p2, dim=dim,
                  p1_mapping=p1_mapping, p2_mapping=p2_mapping)


def eval_func(f):
    #@lru_cache(None)
    @wraps(f)
    def wrapper(obj, params=None):
        if isinstance(params, tuple):
            pass
        elif isinstance(params, list):
            params = tuple(params)
        elif params is None:
            params = ()
        else:
            params = (params, )
        return f(obj, params)
    return wrapper


def vector_func(g):
    #@lru_cache(None)
    @wraps(g)
    def wrapper(obj, params=None, i=None):
        if isinstance(params, tuple):
            pass
        elif isinstance(params, list):
            params = tuple(params)
        elif params is None:
            params = ()
        else:
            params = (params, )
        if i is not None:
            return g(obj, params, i)
        return [g(obj, params, k) for k in range(len(params))]
    return wrapper


def matrix_func(h):
    @wraps(h)
    def wrapper(obj, params=None, i=None, j=None):
        if isinstance(params, tuple):
            pass
        elif isinstance(params, list):
            params = tuple(params)
        elif params is None:
            params = ()
        else:
            params = (params, )
        if i is not None and j is not None:
            return h(obj, params, i, j)
        return [[h(obj, params, k, l)
                 for l in range(k + 1)]
                for k in range(len(params))]
    return wrapper


def not_implemented_foo(obj, *args, **kwargs):
    raise NotImplementedError


class FuncMeta(type):

    def __new__(cls, name, bases, attrs, **kwargs):
        attrs['eval'] = eval_func(attrs['eval'])
        if 'grad' in attrs:
            grad = vector_func(attrs['grad'])
        else:
            grad = not_implemented_foo
        if 'hess' in attrs:
            hess = matrix_func(attrs['hess'])
        else:
            hess = not_implemented_foo
        attrs['grad'] = grad
        attrs['hess'] = hess
        return type.__new__(cls, name, bases, attrs, **kwargs)


class Func(with_metaclass(FuncMeta, object)):

    def eval(self, params):
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

        if isnull(base.grad):
            self.grad = null_method(self)

        if isnull(base.hess):
            self.hess = null_method(self)

        if isinstance(base, Affine):
            self.base = base.base
            self.a = a * base.a
            self.b = a * base.b + b
        else:
            self.base = base
            self.a = a
            self.b = b

    def eval(self, params):
        return self.a * self.base.eval(params) + self.b

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

        if isnull(f.grad):
            self.grad = null_method(self)
            if isnull(f.hess):
                self.hess = null_method(self)

        elif isnull(f.hess):
            def hess(obj, params, i, j):
                f_arg = obj.__f_arg(params)
                h_val = Tensor(0)
                for k, g in enumerate(obj.g_list):
                    h_val += obj.f.grad(f_arg, k).dot_right(
                             g.hess(params, i, j).drop_dim())
                return h_val
            
            self.hess = MethodType(matrix_func(hess), self)

    def __f_arg(self, params):
        return tuple([g.eval(params) for g in self.g_list])

    def eval(self, params):
        return self.f.eval(self.__f_arg(params))

    def grad(self, params, i):
        f_arg = self.__f_arg(params)
        return sum([self.f.grad(f_arg, k).\
                    dot_left(g.grad(params, i).drop_dim())
                    for k, g in enumerate(self.g_list)])

    def hess(self, params, i, j):
        f_arg = self.__f_arg(params)
        h_val = Tensor(0)
        dg_i = []
        dg_j = []

        for k, g in enumerate(self.g_list):
            dg_i.append(g.grad(params, i).drop_dim())

        if i != j:
            for k, g in enumerate(self.g_list):
                dg_j.append(g.grad(params, j).drop_dim())
        else:
            dg_j = dg_i

        for k, g_k in enumerate(self.g_list):
            for l, g_l in enumerate(self.g_list):
                f_hess = self.f.hess(f_arg, k, l)
                h_val += f_hess.dot_left(dg_i[k]).transpose().\
                                dot_left(dg_j[l]).transpose()
            f_grad = self.f.grad(f_arg, k)
            g_hess = g_k.hess(params, i, j).drop_dim()
            h_val += f_grad.dot_right(g_hess)
        return h_val
