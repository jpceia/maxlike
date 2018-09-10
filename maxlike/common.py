from __future__ import print_function
import numpy as np
from hashlib import sha1
from scipy.special import factorial
from random import getrandbits


class Param(np.ma.MaskedArray):
    def __new__(cls, data, *args, **kwargs):
        obj = super(Param, cls).__new__(cls, data, *args, **kwargs)
        obj.hash = getrandbits(128)
        return obj

    def reset(self, data, mask=None):
        self.data[:] = data
        if mask is not None:
            self.mask = mask

    def __hash__(self):
        return self.hash


class IndexMap(list):
    def __init__(self, indexes):
        if isinstance(indexes, int):
            indexes = [indexes]
        # assert min(indexes) >= 0
        self.extend(indexes)

    def __call__(self, params):
        return tuple([params[k] for k in self])
