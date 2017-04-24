from functools import wraps


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


def transpose(result, params, i, j):
    slc = [slice(None)] * params[i].ndim
    slc += [None] * params[j].ndim
    slc += [Ellipsis]
    return result[i][slc]


class IndexMap(list):
    def __init__(self, indexes):
        if isinstance(indexes, int):
            indexes = [indexes]
        assert min(indexes) >= 0
        self.extend(indexes)

    def __call__(self, params):
        return map(params.__getitem__, self)
