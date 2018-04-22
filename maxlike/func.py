import numpy as np
from tensor import *
from common import *


class Func(object):
    def __call__(self, params):
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


# maybe this can be replaced by a metaclass
class Atom(Func):
    def __init__(self, ndim, param_map, feat_map, foo):
        assert isinstance(foo, Func)
        # assert max(feat_map) <= ndim
        self.ndim = ndim
        self.param_map = IndexMap(param_map)
        self.feat_map = feat_map
        self.foo = foo

    def __call__(self, params):
        return self.foo(self.param_map(params)).\
            expand(self.feat_map, self.ndim)

    def grad(self, params, i):
        try:
            idx = self.param_map.index(i)
        except ValueError:
            return Tensor()
        else:
            return self.foo.grad(
                self.param_map(params), idx).expand(
                    self.feat_map, self.ndim)

    def hess(self, params, i, j):
        try:
            idx = self.param_map.index(i)
            jdx = self.param_map.index(j)
        except ValueError:
            return Tensor()
        else:
            return self.foo.hess(
                self.param_map(params), idx, jdx).expand(
                    self.feat_map, self.ndim)


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

    @property
    def param_feat(self):
        return self.base.param_feat

    @call_func
    def __call__(self, params):
        return self.a * self.base(params) + self.b

    @vector_func
    def grad(self, params, i):
        return self.a * self.base.grad(params, i)

    @matrix_func
    def hess(self, params, i, j):
        return self.a * self.base.hess(params, i, j)


class Sum(Func):
    def __init__(self, ndim):
        self.atoms = []
        self.ndim = ndim
        self.b = 0

    def add(self, param_map, feat_map, foo, weight=1.0):
        """
        Adds a factor to Ensemble object.

        Parameters
        ----------
        param_map: int, list
            index of parameters that 'foo' accepts.
        feat_map:
            index of features that 'foo' returns.
        foo: Func
            function to add to the Ensemble object.
        weight: double
            weight
        """
        if isinstance(param_map, int):
            param_map = [param_map]
        if isinstance(feat_map, int):
            feat_map = [feat_map]

        if isinstance(foo, Affine):
            self.b += foo.b
            self.add(param_map, feat_map, foo.base, weight * foo.a)
        elif isinstance(foo, Sum):
            self.b += foo.b
            for w, atom in foo.atoms:
                self.add(
                    atom.param_map(param_map),
                    atom.feat_map(feat_map),
                    atom.foo,
                    w * weight)
        else:
            self.atoms.append((
                weight, Atom(self.ndim, param_map, feat_map, foo)))
        return self

    @call_func
    def __call__(self, params):
        return sum((w * atom(params) for w, atom in self.atoms)) + self.b

    @vector_func
    def grad(self, params, i):
        return sum((w * atom.grad(params, i) for w, atom in self.atoms))

    @matrix_func
    def hess(self, params, i, j):
        return sum((w * atom.hess(params, i, j) for w, atom in self.atoms))


class Linear(Func):
    def __init__(self, weight_list=None):
        if weight_list is None:
            weight_list = []
        elif not isinstance(weight_list, (list, tuple)):
            weight_list = [weight_list]
        self.weights = weight_list

    @call_func
    def __call__(self, params):
        return sum(((w * np.asarray(p)).sum()
                    for w, p in zip(params, self.weights)))

    @vector_func
    def grad(self, params, i):
        return grad_tensor(self.weights[i] * np.ones(params[i].shape),
                           params, i)

    @matrix_func
    def hess(self, params, i, j):
        return hess_tensor(np.multiply.outer(
            np.zeros(params[j].shape),
            np.zeros(params[i].shape)),
            params, i, j)

    def add_feature(self, weight):
        self.weights.append(weight)


class Quadratic(Func):  # TODO : expand this class to allow more generic stuff
    def __init__(self, u=0, s=1):
        self.u = u
        self.s = s

    @call_func
    def __call__(self, params):
        val = 0
        for p in params:
            z = (np.asarray(p) - self.u) / self.s
            val += (z * z).sum()
        return Tensor(val)

    @vector_func
    def grad(self, params, i):
        z = (np.asarray(params[i]) - self.u) / self.s
        return grad_tensor(2 * z / self.s, params, i, True).sum()

    @matrix_func
    def hess(self, params, i, j):
        if i != j:
            return Tensor()
        return hess_tensor(
            2 * np.ones(params[i].shape) / self.s / self.s,
            params, i, j, True, True).sum()


class Encode(Func):
    @call_func
    def __call__(self, params):
        return Tensor(params[0])

    @vector_func
    def grad(self, params, i):
        return grad_tensor(np.ones(params[0].shape), params, i, True)

    @matrix_func
    def hess(self, params, i, j):
        return hess_tensor(np.zeros(params[0].shape), params, i, j, True, True)


class Constant(Func):
    def __init__(self, vector):
        self.vector = np.asarray(vector)

    @call_func
    def __call__(self, params):
        return Tensor(self.vector)

    @vector_func
    def grad(self, params, i):
        return Tensor(np.zeros(self.vector.shape))

    @matrix_func
    def hess(self, params, i, j):
        return Tensor(np.zeros(self.vector.shape))


class Vector(Func):
    def __init__(self, vector):
        self.vector = np.asarray(vector)

    @call_func
    def __call__(self, params):
        return Tensor(np.asarray(params[0]) * self.vector)

    @vector_func
    def grad(self, params, i):
        return grad_tensor(self.vector, params, i, None)

    @matrix_func
    def hess(self, params, i, j):
        return hess_tensor(np.zeros(self.vector.shape),
                           params, i, j, None, None)


class Exp(Func):

    def __init__(self, size=10):
        super(Exp, self).__init__()

    @call_func
    def __call__(self, params):
        return Tensor(np.exp(np.asarray(params[0])))

    @vector_func
    def grad(self, params, i):
        return grad_tensor(np.exp(np.asarray(params[0])), params, i, True)

    @matrix_func
    def hess(self, params, i, j):
        return hess_tensor(np.exp(np.asarray(params[0])),
                           params, i, j, True, True)


class PoissonVector(Func):

    def __init__(self, size=10):
        self.size = size

    @call_func
    def __call__(self, params):
        a = np.asarray(params[0])
        rng = np.arange(self.size)
        return Tensor(np.exp(-a)[..., None] * (a[..., None] ** rng) /
                      factorial(rng), e=1)

    @vector_func
    def grad(self, params, i):
        vec = self(params).values
        return grad_tensor(np.insert(vec[..., :-1], 0, 0, -1) - vec,
                           params, i, True, e=1)

    @matrix_func
    def hess(self, params, i, j):
        vec = self.grad(params, i).values[0]
        return hess_tensor(np.insert(vec[..., :-1], 0, 0, -1) - vec,
                           params, i, j, True, True, e=1)


class GaussianCopula(Func):
    """
    Receives the marginals of X and Y and returns
    a joint distribution of XY using a gaussian copula
    """

    def __init__(self, rho):
        self.rho = rho

    @staticmethod
    def _normalize(x):
        x = ndtri(np.asarray(x).cumsum(-1))
        x[x == np.inf] = 999
        return x

    @call_func
    def __call__(self, params):
        X = self._normalize(params[0])
        Y = self._normalize(params[1])
        F_xy = gauss_bivar(X[..., None, :], Y[..., None], self.rho)
        F_xy = np.insert(F_xy, 0, 0, -1)
        F_xy = np.insert(F_xy, 0, 0, -2)
        return Tensor(np.diff(np.diff(F_xy, 1, -1), 1, -2))


class IndependenceCopula(Func):
    @call_func
    def __call__(self, params):
        F_x = np.asarray(params[0]).cumsum(-1)
        F_y = np.asarray(params[1]).cumsum(-1)
        F_xy = F_x[..., None, :] * F_y[..., None]
        F_xy = np.insert(F_xy, 0, 0, -1)
        F_xy = np.insert(F_xy, 0, 0, -2)
        return Tensor(np.diff(np.diff(F_xy, 1, -1), 1, -2))


class CollapseMatrix(Func):

    def __init__(self, conditions=None):
        """
        Condition or list of conditions with the form
        sgn(A*x + B*y + C) == s
        """
        if conditions is None:
            self.conditions = [
                (1, -1, 0, -1),
                (1, -1, 0, 0),
                (1, -1, 0, 1),
            ]

    @call_func
    def __call__(self, params):
        frame = np.asarray(params[0])
        rng_x = np.arange(frame.shape[-2])
        rng_y = np.arange(frame.shape[-1])
        val = []
        for x, y, c, s in self.conditions:
            filt = np.sign(x * rng_x[:, None] +
                           y * rng_y[None, :] + c) == s
            val.append((frame * filt).sum((-1, -2)))
        return Tensor(np.asarray(val).swapaxes(-1, -2))


class Compose(Func):

    def __init__(self, f, g_list):
        if not isinstance(g_list, (list, tuple)):
            self.g_list = [g_list]
        else:
            self.g_list = g_list
        self.f = f

    def __f_arg(self, params):
        return [g(params) for g in self.g_list]

    @call_func
    def __call__(self, params):
        return self.f(self.__f_arg(params))

    @vector_func
    def grad(self, params, i):
        f_arg = self.__f_arg(params)
        return sum([self.f.grad(f_arg, k).dot(g.grad(params, i))
                    for k, g in enumerate(self.g_list)])

    @matrix_func
    def hess(self, params, i, j):
        f_arg = self.__f_arg(params)
        h_val = 0
        for k, g_k in enumerate(self.g_list):
            for l, g_l in enumerate(self.g_list):
                h_val += self.f.hess(f_arg, k, l).\
                    dot(g_k.grad(params, i)).dot(g_l.grad(params, j))
            h_val += self.f.grad(f_arg, k).dot(g_k.hess(params, i, j))
        return h_val
