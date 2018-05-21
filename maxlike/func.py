import numpy as np
from six import with_metaclass
try:
    from tensor import *
    from common import *
except:
    from .tensor import *
    from .common import *




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

    def __sub__(self, b):
        return Affine(self, 1, -b)

    def __neg__(self):
        return Affine(self, -1, 0)

    def __mul__(self, a):
        return Affine(self, a, 0)

    def __div__(self, a):
        return Affine(self, 1.0 / a, 0)

    def __matmul__(self, other):
        return Compose(self, other)


class FuncWrap(Func):
    def __init__(self, foo, param_map, feat_map, n_feat=None,
                 dim_map=None, n_dim=0, feat_flip=None):
        assert isinstance(foo, Func)
        # assert max(feat_map) <= n_feat
        # assert max(dim_map) <= n_dim
        self.foo = foo
        self.n_feat = n_feat
        self.n_dim = n_dim
        if param_map is None:
            param_map = []
        self.param_map = IndexMap(param_map)
        if feat_map is None:
            feat_map = []
        self.feat_map = IndexMap(feat_map)
        self.feat_flip = feat_flip
        if dim_map is None:
            self.dim_map = IndexMap(range(n_dim))
        else:
            self.dim_map = IndexMap(dim_map)

    def __call__(self, params):
        return self.foo(self.param_map(params)).\
            expand(self.feat_map, self.n_feat).flip(self.feat_flip).\
            expand(self.dim_map, self.n_dim, dim=True)

    def grad(self, params, i):
        try:
            idx = self.param_map.index(i)
        except ValueError:
            return Tensor()
        else:
            return self.foo.grad(self.param_map(params), idx).\
                expand(self.feat_map, self.n_feat).flip(self.feat_flip).\
                expand(self.dim_map, self.n_dim, dim=True)

    def hess(self, params, i, j):
        try:
            idx = self.param_map.index(i)
            jdx = self.param_map.index(j)
        except ValueError:
            return Tensor()
        else:
            return self.foo.hess(
                self.param_map(params), idx, jdx).\
                    expand(self.feat_map, self.n_feat).flip(self.feat_flip).\
                    expand(self.dim_map, self.n_dim, dim=True)


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

    def __call__(self, params):
        return self.a * self.base(params) + self.b

    def grad(self, params, i):
        return self.a * self.base.grad(params, i)

    def hess(self, params, i, j):
        return self.a * self.base.hess(params, i, j)


class Sum(Func):
    def __init__(self, n_feat, n_dim=0):
        self.atoms = []
        self.n_feat = n_feat
        self.n_dim = n_dim
        self.b = 0

    def add(self, foo, param_map, feat_map, dim_map=None, weight=1.0):
        """
        Adds a factor to Ensemble object.

        Parameters
        ----------
        foo: Func
            function to add to the Ensemble object.
        param_map: int, list
            index of parameters that 'foo' accepts.
        feat_map:
            index of features that 'foo' returns.
        dim_map:
            index of dim of 'foo' image space.
        weight: double
            weight
        """
        if isinstance(param_map, int):
            param_map = [param_map]
        if feat_map is None:
            feat_map = []
        elif isinstance(feat_map, int):
            feat_map = [feat_map]
        if dim_map is None:
            dim_map = []
        elif isinstance(dim_map, int):
            dim_map = [dim_map]

        if isinstance(foo, Affine):
            self.b += foo.b
            self.add(foo.base, param_map, feat_map, dim_map, weight * foo.a)
        elif isinstance(foo, Sum):
            self.b += foo.b
            for w, atom in foo.atoms:
                self.add(
                    atom.foo,
                    atom.param_map(param_map),
                    atom.feat_map(feat_map),
                    atom.dim_map(dim_map),
                    w * weight)
        else:
            self.atoms.append((
                weight, FuncWrap(
                    foo, param_map,
                    feat_map, self.n_feat,
                    dim_map, self.n_dim)))
        return self

    def __call__(self, params):
        return sum((w * atom(params) for w, atom in self.atoms)) + self.b

    def grad(self, params, i):
        return sum((w * atom.grad(params, i) for w, atom in self.atoms))

    def hess(self, params, i, j):
        return sum((w * atom.hess(params, i, j) for w, atom in self.atoms))


class Product(Func):
    def __init__(self, n_feat, n_dim=0):
        self.atoms = []
        self.n_feat = n_feat
        self.n_dim = n_dim

    def add(self, foo, param_map, feat_map, dim_map=None):

        if isinstance(param_map, int):
            param_map = [param_map]
        if isinstance(feat_map, int):
            feat_map = [feat_map]
        if dim_map is None:
            dim_map = []
        elif isinstance(dim_map, int):
            dim_map = [dim_map]

        if isinstance(foo, Product):
            for atom in foo.atoms:
                self.add(
                    atom.foo,
                    atom.param_map(param_map),
                    atom.feat_map(feat_map),
                    atom.dim_map(dim_map))
        else:
            self.atoms.append(FuncWrap(
                foo, param_map,
                feat_map, self.n_feat,
                dim_map, self.n_dim))

        return self

    @staticmethod
    def _prod(arr, except_idx):
        p = 1
        for i, el in enumerate(arr):
            if i not in except_idx:
                p *= el
        return p

    def __call__(self, params):
        return Product._prod([atom(params) for atom in self.atoms], [])

    def grad(self, params, i):
        f_val = [atom(params) for atom in self.atoms]
        grad = 0
        for k, atom in enumerate(self.atoms):
            f_prod = Product._prod(f_val, [k])
            grad += f_prod * atom.grad(params, i)
        return grad

    def hess(self, params, i, j):
        f_val = [atom(params) for atom in self.atoms]
        hess_val = 0
        for k, a_k in enumerate(self.atoms):
            hess_k = 0
            for l, a_l in enumerate(self.atoms):
                if k != l:
                    f_prod = Product._prod(f_val, [k, l])
                    hess_k += f_prod * a_l.grad(params, j).transpose()
            hess_k *= a_k.grad(params, i)
            f_prod = Product._prod(f_val, [k])
            hess_k += f_prod * a_k.hess(params, i, j)
            hess_val += hess_k
        return hess_val


class Linear(Func):
    def __init__(self, weight_list=None):
        if weight_list is None:
            weight_list = []
        elif not isinstance(weight_list, (list, tuple)):
            weight_list = [weight_list]
        self.weights = weight_list

    def __call__(self, params):
        return sum(((w * np.asarray(p)).sum()
                    for w, p in zip(params, self.weights)))

    def grad(self, params, i):
        return grad_tensor(self.weights[i] * np.ones((1, ) * params[i].ndim),
                           params, i)

    def hess(self, params, i, j):
        return hess_tensor(np.zeros((1, )* (params[j].ndim + params[i].ndim)),
            params, i, j)

    def add_feature(self, weight):
        self.weights.append(weight)


class Quadratic(Func):  # TODO : expand this class to allow more generic stuff
    def __init__(self, u=0, s=1):
        self.u = u
        self.s = s

    def __call__(self, params):
        val = 0
        for p in params:
            z = (np.asarray(p) - self.u) / self.s
            val += (z * z).sum()
        return Tensor(val)

    def grad(self, params, i):
        z = (np.asarray(params[i]) - self.u) / self.s
        return grad_tensor(2 * z / self.s, params, i, True).sum()

    def hess(self, params, i, j):
        if i != j:
            return Tensor()
        return hess_tensor(
            2 * np.ones(params[i].shape) / self.s / self.s,
            params, i, j, True, True).sum()


class Encode(Func):

    def __call__(self, params):
        return Tensor(params[0])

    def grad(self, params, i):
        return grad_tensor(np.ones((1, ) * params[0].ndim), params, i, True)

    def hess(self, params, i, j):
        return hess_tensor(np.zeros((1, ) * params[0].ndim), params, i, j, True, True)


class Constant(Func):
    def __init__(self, vector):
        self.vector = np.asarray(vector)

    def __call__(self, params):
        return Tensor(self.vector)

    def grad(self, params, i):
        return Tensor(np.zeros(self.vector.shape))

    def hess(self, params, i, j):
        return Tensor(np.zeros(self.vector.shape))


class Scalar(Func):
    def __call__(self, params):
        return Tensor(params[0])

    def grad(self, params, i):
        return Tensor(1)

    def hess(self, params, i, j):
        return Tensor(0)


class Vector(Func):
    def __init__(self, vector):
        self.vector = np.asarray(vector)

    def __call__(self, params):
        return Tensor(np.asarray(params[0]) * self.vector)

    def grad(self, params, i):
        return grad_tensor(self.vector, params, i, None)

    def hess(self, params, i, j):
        return hess_tensor(np.zeros(self.vector.shape),
                           params, i, j, None, None)


class Exp(Func):

    def __call__(self, params):
        return Tensor(np.exp(np.asarray(params[0])))

    def grad(self, params, i):
        return grad_tensor(np.exp(np.asarray(params[0])), params, i, True)

    def hess(self, params, i, j):
        return hess_tensor(np.exp(np.asarray(params[0])),
                           params, i, j, True, True)


class Poisson(Func):

    """
    Non normalized
    vector (lambda ^ x) / x!
    """

    def __init__(self, size=10):
        self.size = size

    def __call__(self, params):
        a = np.asarray(params[0])
        rng = np.arange(self.size)
        vec = (a[..., None] ** rng) / factorial(rng)
        return Tensor(vec, dim=1)

    def grad(self, params, i):
        a = np.asarray(params[0])
        rng = np.arange(self.size)
        vec = ((a[..., None] ** rng) / factorial(rng))[..., :-1]
        vec = np.insert(vec, 0, 0, -1)
        return grad_tensor(vec, params, i, True, dim=1)

    def hess(self, params, i, j):
        a = np.asarray(params[0])
        rng = np.arange(self.size)
        vec = ((a[..., None] ** rng) / factorial(rng))[..., :-2]
        vec = np.insert(vec, 0, 0, -1)
        vec = np.insert(vec, 0, 0, -1)
        return hess_tensor(vec, params, i, j, True, True, dim=1)


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

    def __call__(self, params):
        X = self._normalize(params[0])
        Y = self._normalize(params[1])
        F_xy = gauss_bivar(X[..., None, :], Y[..., None], self.rho)
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

    def __call__(self, params):
        """
        CollapseMatrix function assumes that there is just one param that is
        a Tensor with dim=2 (frame)
        """
        arr = np.asarray(params[0])
        rng_x = np.arange(arr.shape[-2])
        rng_y = np.arange(arr.shape[-1])
        val = []
        for x, y, c, s in self.conditions:
            filt = np.sign(x * rng_x[:, None] +
                           y * rng_y[None, :] + c) == s
            val.append((arr * filt).sum((-1, -2)))
        val = np.stack(val, -1)
        return Tensor(val, dim=1)

    def grad(self, params, i):
        ones = np.ones(np.asarray(params[0]).shape)
        rng_x = np.arange(ones.shape[-2])
        rng_y = np.arange(ones.shape[-1])
        val = []
        for x, y, c, s in self.conditions:
            filt = np.sign(x * rng_x[:, None] +
                           y * rng_y[None, :] + c) == s
            val.append((ones * filt).sum((-1, -2)))
        val = np.stack(val, -1)
        p1_mapping = list(range(ones.ndim - 2)) + [None, None]
        return grad_tensor(
            val, params, i, p1_mapping=p1_mapping, dim=1)

    def hess(self, params, i, j):
        return Tensor()


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
