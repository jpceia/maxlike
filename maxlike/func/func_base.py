import numpy as np
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


def grad_tensor(values, params, i=0, x_map=None, dim=0):

    x_dim = np.asarray(params[i]).ndim
    if x_map is None:
        idx = [...]
        x_map = array('b')
    else:
        idx = [None] * x_dim + [...]
        if x_map is True:
            x_map = array('b', range(x_dim))
        elif isinstance(x_map, (list, tuple)):
            x_map = array('b', x_map)
        else:
            raise ValueError("Invalid mapping")
        
    return Tensor(np.asarray(values)[tuple(idx)], x_dim=x_dim, dim=dim,
                  x_map=x_map)


def hess_tensor(values, params, i=0, j=0,
                x_map=None, y_map=None, dim=0):
    x_dim = np.asarray(params[i]).ndim
    y_dim = np.asarray(params[j]).ndim
    idx  = [slice(None) if x_map is None else None] * x_dim
    idx += [slice(None) if y_map is None else None] * y_dim
    idx += [...]

    if x_map is None:
        x_map = array('b')
    else:
        if x_map is True:
            x_map = array('b', range(x_dim))
        elif isinstance(x_map, (list, tuple)):
            x_map = array('b', x_map)
        else:
            raise ValueError("Invalid mapping")

    if y_map is None:
        y_map = array('b')
    else:
        if y_map is True:
            y_map = array('b', range(y_dim))
        elif isinstance(y_map, (list, tuple)):
            y_map = array('b', y_map)
        else:
            raise ValueError("Invalid mapping")

    return Tensor(np.asarray(values)[tuple(idx)], x_dim=x_dim, y_dim=y_dim, dim=dim,
                  x_map=x_map, y_map=y_map)


def as_tuple(params):
    """Normalises the `params` argument of a Func method to a tuple"""
    if isinstance(params, tuple):
        return params
    if isinstance(params, list):
        return tuple(params)
    if params is None:
        return ()
    return (params, )


def call_wrap(f):
    @wraps(f)
    def wrapper(obj, params=None):
        return f(obj, as_tuple(params))
    return wrapper


def vector_wrap(g):
    @wraps(g)
    def wrapper(obj, params=None, i=None):
        params = as_tuple(params)
        if i is not None:
            return g(obj, params, i)
        return [g(obj, params, k) for k in range(len(params))]
    return wrapper


def matrix_wrap(h):
    @wraps(h)
    def wrapper(obj, params=None, i=None, j=None):
        params = as_tuple(params)
        if i is not None and j is not None:
            return h(obj, params, i, j)
        return [[h(obj, params, k, l)
                 for l in range(k + 1)]
                for k in range(len(params))]
    return wrapper


# eval takes the same arguments as __call__
eval_wrap = call_wrap


def not_implemented_foo(obj, *args, **kwargs):
    raise NotImplementedError


class FuncMeta(type):
    """
    Metaclass of Func. It wraps __call__, grad and hess so that they accept
    the parameters as a list and return Tensors: grad without an index
    returns the list of all gradients, hess without indices the
    lower-triangular list of lists of all second derivatives. It also adds
    a default eval (value, gradients and Hessian in one call) when a
    subclass does not define one.
    """

    def __new__(cls, name, bases, attrs, **kwargs):
        call = call_wrap(attrs['__call__'])
        if 'grad' in attrs:
            grad = vector_wrap(attrs['grad'])
        else:
            grad = not_implemented_foo
        if 'hess' in attrs:
            hess = matrix_wrap(attrs['hess'])
        else:
            hess = not_implemented_foo
        if 'eval' not in attrs:
            def eval_func(obj, params):
                return attrs['__call__'](obj, params), \
                       attrs['grad'](obj, params), \
                       attrs['hess'](obj, params)
            attrs['eval'] = eval_wrap(eval_func)
        else:
            attrs['eval'] = eval_wrap(attrs['eval'])
        attrs['__call__'] = call
        attrs['grad'] = grad
        attrs['hess'] = hess
        return type.__new__(cls, name, bases, attrs, **kwargs)


class Func(metaclass=FuncMeta):
    """
    Base class of the functions of the parameters that make up a model.

    A subclass defines

        __call__(params)    the value, a Tensor
        grad(params, i)     derivative with respect to params[i]
        hess(params, i, j)  second derivative with respect to params[i]
                            and params[j] (j <= i)

    and optionally eval(params), which returns all three at once. params is
    a list of arrays. The derivatives are Tensors that keep the structure
    of the function (diagonal in the parameter / feature axes they share).

    Funcs combine with arithmetic and composition: a * f + b (Affine),
    -f, and f @ g (Compose, f applied to the output of g).
    """

    def __call__(self, params):
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

    def graph(self, param_names=None):
        """
        Computation graph of this function (see maxlike.graph); export it
        with .to_dot(), .to_mermaid() or .render(path).
        """
        from ..graph import func_graph
        return func_graph(self, param_names)


class Affine(Func):
    """
    Affine(f, a, b)(x) = a * f(x) + b
    """

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

    def __call__(self, params):
        return self.a * self.base(params) + self.b

    def grad(self, params, i):
        return self.a * self.base.grad(params, i)

    def hess(self, params, i, j):
        return self.a * self.base.hess(params, i, j)

    def eval(self, params):
        base_val, base_grad, base_hess = self.base.eval(params)
        val = self.a * base_val + self.b
        grad = [self.a * d for d in base_grad]
        hess = [[self.a * h for h in h_line] for h_line in base_hess]
        return val, grad, hess


class Compose(Func):
    """
    Composition of Funcs
    Compose(f, [g1, ..., gn])(x) := f(g1(x), ..., gn(x))
    """

    def __init__(self, f, g_list):

        if not isinstance(g_list, (list, tuple)):
            self.g_list = [g_list]
        else:
            self.g_list = g_list

        self.f = f

        if isnull(f.grad):
            self.grad = null_method(self)
            self.hess = null_method(self)

    def __call__(self, params):
        g_val = [g(params) for g in self.g_list]
        return self.f(g_val)

    def grad(self, params, i):
        g_val = [g(params) for g in self.g_list]
        return sum([self.f.grad(g_val, k).\
                    dot_left(g.grad(params, i).drop_dim())
                    for k, g in enumerate(self.g_list)])

    def hess(self, params, i, j):
        g_val = [g(params) for g in self.g_list]
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
                if l > k:
                    f_hess = self.f.hess(g_val, l, k).transpose()
                else:
                    f_hess = self.f.hess(g_val, k, l)
                h_val += f_hess.dot_left(dg_i[k]).transpose().\
                                dot_left(dg_j[l]).transpose()
            f_grad = self.f.grad(g_val, k)
            g_hess = g_k.hess(params, i, j).drop_dim()
            h_val += f_grad.dot_right(g_hess)
        return h_val


    def eval(self, params):
        g_val, g_grad, g_hess = zip(*(g.eval(params) for g in self.g_list))
        g_grad = [[d.drop_dim() for d in grad] for grad in g_grad]
        val, f_grad, f_hess = self.f.eval(g_val)
        grad, hess = [], []

        def f_hess_kl(k, l):
            return f_hess[l][k].transpose() if l > k else f_hess[k][l]

        for i in range(len(params)):

            grad.append(sum((
                f_grad[k].dot_left(d_k[i])
                for k, d_k in enumerate(g_grad))))

            # first contraction of H_f, shared by the whole row of blocks
            hf_i = [[f_hess_kl(k, l).dot_left(d_k[i]).transpose()
                     for l in range(len(g_grad))]
                    for k, d_k in enumerate(g_grad)]

            hess_line = []
            for j in range(i + 1):
                hess_ij = Tensor(0)
                for k, h_k in enumerate(g_hess):
                    hess_ij += f_grad[k].dot_right(h_k[i][j].drop_dim())
                    for l, d_l in enumerate(g_grad):
                        hess_ij += hf_i[k][l].dot_left(d_l[j]).transpose()
                hess_line.append(hess_ij)
            hess.append(hess_line)
        return val, grad, hess
