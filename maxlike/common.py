from __future__ import print_function
import numpy as np
from hashlib import sha1
from scipy.special import factorial


class Param(np.ma.MaskedArray):
    def reset(self, data, mask=None):
        self.data[:] = data
        if mask is not None:
            self.mask = mask

    def __hash__(self):
        return int(sha1(self.data).hexdigest(), 16)


class IndexMap(list):
    def __init__(self, indexes):
        if isinstance(indexes, int):
            indexes = [indexes]
        # assert min(indexes) >= 0
        self.extend(indexes)

    def __call__(self, params):
        return tuple([params[k] for k in self])
