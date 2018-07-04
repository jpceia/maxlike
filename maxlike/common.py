from __future__ import print_function
import numpy as np
from hashlib import sha1
from scipy.special import factorial


class Params(list):
    def __hash__(self):
        return hash(tuple(self))


class Param(np.ma.MaskedArray):
    def reset(self, data, mask=None):
        self.data[:] = data
        if mask is not None:
            self.mask = mask

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


class IndexMap(list):
    def __init__(self, indexes):
        if isinstance(indexes, int):
            indexes = [indexes]
        # assert min(indexes) >= 0
        self.extend(indexes)

    def __call__(self, params):
        return Params([params[k] for k in self])
