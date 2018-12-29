from ..tensor import Tensor
from .func_base import Func, Affine


class IndexMap(list):
    def __init__(self, indexes):
        if isinstance(indexes, int):
            indexes = [indexes]
        # assert min(indexes) >= 0
        self.extend(indexes)

    def __call__(self, params):
        return tuple([params[k] for k in self])


class FuncWrap(Func):
    def __init__(self, foo, param_map, feat_map=None, n_feat=None,
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
            n_dim = 0
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
                    hess_k += f_prod * a_k.grad(params, i) * a_l.grad(params, j).transpose()
            hess_val += hess_k
            hess_val += Product._prod(f_val, [k]) * a_k.hess(params, i, j)
        return hess_val
