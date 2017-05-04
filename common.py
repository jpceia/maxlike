import numpy as np
from functools import wraps


class CacheWrap:
    def __init__(self, foo):
        self.foo = foo
        self.params = None
        self.args = []
        self.kwargs = dict()

    def __call__(self, params):
        n = len(params)
        cached = True
        if self.params is None:
            cached = False
        elif len(self.params) != n:
            cached = False
        else:
            for k in range(n):
                if not cached:
                    break
                if self.params[k].shape != params[k].shape:
                    cached = False
                elif any(self.params[k] != params[k]):
                    cached = False
        if not cached:
            self.params = list(params)
            self.cached_result = self.foo(params)
        return self.cached_result


def call_func(f):
    @wraps(f)
    def wrapper(obj, params=None):
        if params is None:
            params = []
        return f(obj, params)
    return wrapper


def vector_func(g):
    @wraps(g)
    def wrapper(obj, params=None, i=None, diag=False):
        if params is None:
            params = []
        if i is not None:
            return g(obj, params, i, diag=diag)
        else:
            return map(lambda k: g(obj, params, k, diag=diag),
                       range(len(params)))
    return wrapper


def matrix_func(h):
    @wraps(h)
    def wrapper(obj, params=None, i=None, j=None, diag_i=False, diag_j=False):
        if params is None:
            params = []
        if i is not None and j is not None:
            return h(obj, params, i, j, diag_i=diag_i, diag_j=diag_j)
        else:
            return map(lambda k:
                       map(lambda l:
                           h(obj, params, k, l, diag_i=diag_i, diag_j=diag_j),
                           range(k + 1)), range(len(params)))
    return wrapper


def vector_sum(vector, params, param_feat, i):
    param_feat = param_feat.get(i, [])
    p = params[i].ndim
    for k in range(len(param_feat)):
        vector = vector.swapaxes(k, p + param_feat[k])
    return vector.sum(tuple(np.arange(p, vector.ndim)))


def matrix_sum(matrix, params, param_feat, i, j):
    param_feat_i = param_feat.get(i, [])
    param_feat_j = param_feat.get(j, [])
    pdim_i = params[i].ndim
    pdim_j = params[j].ndim
    for k in range(pdim_i):
        matrix = matrix.swapaxes(
            k + pdim_j,
            param_feat_i[k] + pdim_i + pdim_j)
    for k in range(pdim_j):
        f = param_feat_j[k]
        if f in param_feat_i:
            idx = np.zeros(matrix.ndim, dtype=np.bool)
            idx[param_feat_j.index(f)] = True
            idx[pdim_j + k] = True
            idx = map(lambda b: slice(None) if b else None, idx)
            matrix = matrix * np.eye(matrix.shape[pdim_j + k])[idx]
        else:
            matrix = matrix.swapaxes(k, f + pdim_i + pdim_j)
    return matrix.sum(tuple(np.arange(pdim_i + pdim_j, matrix.ndim)))


def transpose(result, params, i, j):
    slc = [slice(None)] * params[i].ndim
    slc += [None] * params[j].ndim
    slc += [Ellipsis]
    return result[i][slc]


class IndexMap(list):
    def __init__(self, indexes):
        if isinstance(indexes, int):
            indexes = [indexes]
        # assert min(indexes) >= 0
        self.extend(indexes)

    def __call__(self, params):
        return map(params.__getitem__, self)
