from ..tensor import Tensor
from .func_base import Func, Affine, null_method, isnull
from .func import X, Constant, Scalar, Vector
from six.moves import reduce
from types import MethodType
from array import array


class IndexMap(list):

    def __init__(self, indexes):
        # assert min(indexes) >= 0
        try:
            self.extend(indexes)
        except TypeError:
            assert isinstance(indexes, int)
            self.append(indexes)

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

        if isnull(foo.grad):
            self.grad = null_method(self)

        if isnull(foo.hess):
            def eval(obj, params):
                foo_val, foo_grad, foo_hess = foo.eval(obj.param_map(params))
                val = obj._expand(foo_val)
                n = len(params)
                grad = []
                hess = [[Tensor(0)] * (i + 1) for i in range(n)]
                for i in range(n):
                    try:
                        grad.append(obj._expand(foo_grad[obj.param_map.index(i)]))
                    except ValueError:
                        grad.append(Tensor(0))
                return val, grad, hess

            self.hess = null_method(self)
            self.eval = MethodType(eval, self)

    def _expand(self, x):
        return x.expand(self.feat_map, self.n_feat).flip(self.feat_flip).\
                 expand(self.dim_map, self.n_dim, dim=True)

    def __call__(self, params):
        return self._expand(self.foo(self.param_map(params)))

    def grad(self, params, i):
        try:
            return self._expand(
                self.foo.grad(self.param_map(params),
                              self.param_map.index(i)))
        except ValueError:
            return Tensor(0)

    def hess(self, params, i, j):
        try:
            return self._expand(
                self.foo.hess(self.param_map(params),
                              self.param_map.index(i),
                              self.param_map.index(j)))
        except ValueError:
            return Tensor(0)

    def eval(self, params):
        foo_val, foo_grad, foo_hess = self.foo.eval(self.param_map(params))
        val = self._expand(foo_val)
        grad, hess = [], []
        idx_list = []
        for i in range(len(params)):
            try:
                idx = self.param_map.index(i)
                grad.append(self._expand(foo_grad[idx]))
                hess_line = []
                for j in range(i):
                    jdx = idx_list[j]
                    hess_line.append(
                        Tensor(0) if jdx is None else
                        self._expand(foo_hess[idx][jdx]))
                hess_line.append(self._expand(foo_hess[idx][idx]))  # for diag
                hess.append(hess_line)
            except ValueError:
                idx = None
                grad.append(Tensor(0))
                hess.append([Tensor(0)] * (i + 1))
            idx_list.append(idx)
        return val, grad, hess


class Collapse(Func):

    def __init__(self, foo, sum_val=True, sum_dim=True):
        self.foo = foo
        self.sum_val = sum_val
        self.sum_dim = sum_dim

    def __call__(self, params):
        return self.foo(params).\
            sum(sum_val=self.sum_val,
                sum_dim=self.sum_dim)

    def grad(self, params, i):
        return self.foo.grad(params, i).\
            sum(sum_val=self.sum_val,
                sum_dim=self.sum_dim)

    def hess(self, params, i, j):
        return self.foo.hess(params, i, j).\
            sum(sum_val=self.sum_val,
                sum_dim=self.sum_dim)

    def eval(self, params):
        val, grad, hess = self.foo.eval(params)
        return val.sum(self.sum_val, self.sum_dim), \
               [d.sum(self.sum_val, self.sum_dim) for d in grad], \
               [[h.sum(self.sum_val, self.sum_dim)
                 for h in h_line] for h_line in hess]


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

    def eval(self, params):
        n = len(params)
        val = Tensor(0)
        grad = [Tensor(0)] * n
        hess = [[Tensor(0)] * (i + 1) for i in range(n)]
        for w, atom in self.atoms:
            atom_val, atom_grad, atom_hess = atom.eval(params)
            val += w * atom_val
            for i in range(n):
                grad[i] += w * atom_grad[i]
                for j in range(i + 1):
                    hess[i][j] += w * atom_hess[i][j]
        return val, grad, hess


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
        try:
            return reduce(lambda x, y: x * y,
                          (el for i, el in enumerate(arr)
                           if i not in except_idx))
        except TypeError as err:
            return 1.0

    def __call__(self, params):
        return reduce(lambda x, y: x * y,
                      (atom(params) for atom in self.atoms))

    def grad(self, params, i):
        f_val = [atom(params) for atom in self.atoms]
        return sum((Product._prod(f_val, [k]) * atom.grad(params, i)
                    for k, atom in enumerate(self.atoms)))

    def hess(self, params, i, j):
        f_val = [atom(params) for atom in self.atoms]
        hess = 0
        for k, a_k in enumerate(self.atoms):
            hess += sum((Product._prod(f_val, [k, l]) *
                         a_k.grad(params, i) *
                         a_l.grad(params, j).transpose()
                         for l, a_l in enumerate(self.atoms) if k != l))
            hess += Product._prod(f_val, [k]) * a_k.hess(params, i, j)
        return hess

    def eval(self, params):
        n = len(params)
        val = 1
        grad = [Tensor(0)] * n
        hess = [[Tensor(0)] * (i + 1) for i in range(n)]
        for k, atom in enumerate(self.atoms):
            atom_val, atom_grad, atom_hess = atom.eval(params)
            for i in range(n):
                for j in range(i + 1):
                    hess[i][j] = val * atom_hess[i][j] + \
                                 atom_val * hess[i][j] + \
                                 grad[i] * atom_grad[j].transpose() + \
                                 atom_grad[i] * grad[j].transpose()
            for i in range(n):
                grad[i] = val * atom_grad[i] + atom_val * grad[i]
            val *= atom_val
        return val, grad, hess
