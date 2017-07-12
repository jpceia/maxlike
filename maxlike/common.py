import numpy as np
from functools import wraps
from hashlib import sha1


class Params(list):
    def __hash__(self):
        return hash(tuple(self))


class Param(np.ma.masked_array):
    def __hash__(self):
        return int(sha1(self.data).hexdigest(), 16)


class cache_output:
    def __init__(self, foo):
        self.foo = foo
        self.hash = None

    def __call__(self, params, **kwargs):
        hash_val = hash(tuple(params, frozenset(kwargs.items())))
        if self.hash is None or self.hash != hash_val:
            self.hash = hash_val
            self.cached_result = self.foo(params)
        return self.cached_result


def call_func(f):
    @wraps(f)
    def wrapper(obj, params=None, **kwargs):
        if params is None:
            params = []
        return f(obj, params, **kwargs)
    return wrapper


def vector_func(g):
    @wraps(g)
    def wrapper(obj, params=None, i=None,
                diag=False, **kwargs):
        if params is None:
            params = []
        if i is not None:
            return g(obj, params, i, diag=diag, **kwargs)
        return [g(obj, params, k, diag=diag, **kwargs)
                for k in range(len(params))]
    return wrapper


def matrix_func(h):
    @wraps(h)
    def wrapper(obj, params=None, i=None, j=None,
                diag_i=False, diag_j=False, **kwargs):
        if params is None:
            params = []
        if i is not None and j is not None:
            return h(obj, params, i, j, diag_i=diag_i, diag_j=diag_j, **kwargs)
        return [[h(obj, params, k, l, diag_i=diag_i, diag_j=diag_j, **kwargs)
                 for l in range(k + 1)]
                for k in range(len(params))]
    return wrapper


def vector_sum(vector, params, param_feat, i):
    param_feat = param_feat.get(i, [])
    p = params[i].ndim
    for k, f in enumerate(param_feat):
        vector = vector.swapaxes(k, p + f)
    return vector.sum(tuple(np.arange(p, vector.ndim)))


def matrix_sum(matrix, params, param_feat, i, j):
    param_feat_i = param_feat.get(i, [])
    param_feat_j = param_feat.get(j, [])
    p_i = params[i].ndim
    p_j = params[j].ndim
    for k, f in enumerate(param_feat_i):  # range(p_i)
        matrix = matrix.swapaxes(p_j + k, p_i + p_j + f)
    for k, f in enumerate(param_feat_j):  # range(p_j)
        if f in param_feat_i:
            idx = np.zeros(matrix.ndim, dtype=np.bool)
            idx[param_feat_j.index(f)] = True
            idx[p_j + k] = True
            idx = [slice(None) if x else None for x in idx]
            matrix = matrix * np.eye(matrix.shape[p_j + k])[idx]
        else:
            matrix = matrix.swapaxes(k, p_i + p_j + f)
    return matrix.sum(tuple(np.arange(p_i + p_j, matrix.ndim)))


def transpose(vector, params, i, j):
    if params[i].ndim > vector[i].ndim:
        return np.zeros(())
    slc = [slice(None)] * params[i].ndim
    slc += [None] * params[j].ndim
    slc += [Ellipsis]
    return vector[i][slc]


class IndexMap(list):
    def __init__(self, indexes):
        if isinstance(indexes, int):
            indexes = [indexes]
        # assert min(indexes) >= 0
        self.extend(indexes)

    def __call__(self, params):
        return Params([params[k] for k in self])
