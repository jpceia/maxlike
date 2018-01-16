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
        elif not isinstance(params, (tuple, list)):
            params = [params]
        return f(obj, params, **kwargs)
    return wrapper


def vector_func(g):
    @wraps(g)
    def wrapper(obj, params=None, i=None, **kwargs):
        if params is None:
            params = []
        elif not isinstance(params, (tuple, list)):
            params = [params]
        if i is not None:
            return g(obj, params, i, **kwargs)
        return [g(obj, params, k, **kwargs) for k in range(len(params))]
    return wrapper


def matrix_func(h):
    @wraps(h)
    def wrapper(obj, params=None, i=None, j=None, **kwargs):
        if params is None:
            params = []
        elif not isinstance(params, (tuple, list)):
            params = [params]
        if i is not None and j is not None:
            return h(obj, params, i, j, **kwargs)
        return [[h(obj, params, k, l, **kwargs)
                 for l in range(k + 1)]
                for k in range(len(params))]
    return wrapper


class IndexMap(list):
    def __init__(self, indexes):
        if isinstance(indexes, int):
            indexes = [indexes]
        # assert min(indexes) >= 0
        self.extend(indexes)

    def __call__(self, params):
        return Params([params[k] for k in self])
