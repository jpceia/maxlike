
def call_func(f):
    def wrapper(obj, params=None):
        if params is None:
            params = []
        return f(obj, params)
    return wrapper


def vector_func(g):
    def wrapper(obj, params=None, i=None, expand=False):
        if params is None:
            params = []
        if i is not None:
            return g(obj, params, i)
        else:
            return map(lambda k: g(obj, params, k), range(len(params)))
    return wrapper


def matrix_func(h):
    def wrapper(obj, params=None, i=None, j=None):
        if params is None:
            params = []
        if i is not None and j is not None:
            return h(obj, params, i, j)
        else:
            return map(lambda k: map(lambda l: h(obj, params, k, l),
                       range(k + 1)), range(len(params)))
    return wrapper


class IndexMap(list):
    def __init__(self, indexes):
        if isinstance(indexes, int):
            indexes = [indexes]
        assert min(indexes) >= 0
        self.extend(indexes)

    def __call__(self, params):
        return map(params.__getitem__, self)
