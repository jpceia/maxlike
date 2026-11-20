"""
Exhaustive check of the structured Tensor algebra against dense arrays.

Enumerates Tensor structures (number of axes of each kind and every
injective diagonal map, including -1 entries) and compares +, -, *, sum,
transpose, dot_left and dot_right with the same operation done on the
dense arrays returned by toarray().
"""
import itertools
import unittest
import numpy as np
from array import array
from maxlike.tensor import Tensor

SIZE = 3
LETTERS = "abcdefghij"


def maps(k, n):
    """Injective partial maps of k parameter axes into n feature axes."""
    for m in itertools.product(list(range(n)) + [-1], repeat=k):
        used = [f for f in m if f >= 0]
        if len(used) == len(set(used)):
            yield array('b', m)


def structures(max_x=1, max_y=1, ns=(1, 2), dims=(0, 1)):
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            for n in ns:
                for d in dims:
                    for xm in (maps(x, n) if x else [array('b')]):
                        for ym in (maps(y, n) if y else [array('b')]):
                            yield x, y, n, d, xm, ym


def make(rng, x, y, n, d, xm, ym, normalize):
    # mapped parameter axes are stored with size 1 (diagonal)
    shape = [1 if xm[k] >= 0 else SIZE for k in range(x)]
    shape += [1 if ym[k] >= 0 else SIZE for k in range(y)]
    shape += [SIZE] * (n + d)
    if normalize:
        xm = xm if any(f >= 0 for f in xm) else None
        ym = ym if any(f >= 0 for f in ym) else None
    return Tensor(rng.uniform(.5, 1.5, shape), x_dim=x, y_dim=y, dim=d,
                  x_map=xm or None, y_map=ym or None)


def dense(t):
    return np.asarray(t.toarray() if hasattr(t, "toarray") else t, float)


def add_dim(arr, d, target):
    # a tensor without distribution axes broadcasts over the trailing ones
    return arr[(...,) + (None,) * (target - d)] * np.ones((SIZE,) * (target - d))


class Test(unittest.TestCase):

    def assertDense(self, result, expected, desc):
        result, expected = np.broadcast_arrays(dense(result), expected)
        np.testing.assert_allclose(result, expected, err_msg=str(desc))

    def run_all(self, normalize):
        rng = np.random.RandomState(0)

        def mk(*s):
            return make(rng, *s, normalize=normalize)

        # binary operations between tensors with the same axes
        ops = [("add", lambda u, v: u + v),
               ("sub", lambda u, v: u - v),
               ("mul", lambda u, v: u * v)]
        for s1 in structures():
            a = mk(*s1)
            for s2 in structures():
                if s2[:3] != s1[:3] or s2[3] not in (0, s1[3]):
                    continue
                b = mk(*s2)
                for name, op in ops:
                    expected = op(dense(a), add_dim(dense(b), s2[3], s1[3]))
                    self.assertDense(op(a, b), expected, (name, s1, s2))

        # sum over feature and distribution axes, transpose
        for s in structures(max_x=2):
            x, y, n, d = s[:4]
            a = mk(*s)
            A = dense(a)
            self.assertDense(a.sum(True, True),
                             A.sum(tuple(range(x + y, A.ndim))), ("sum", s))
            expected = np.moveaxis(A, list(range(x, x + y)), list(range(y)))
            self.assertDense(a.transpose(), expected, ("transpose", s))

        # dot_left: other[X, N, E] . self[N, Y, M, E] -> [X, Y, M, E]
        for so in structures(max_y=0):
            ox, _, on, od = so[:4]
            other = mk(*so)
            for ss in structures(max_x=2):
                sx, sy, sn, sd = ss[:4]
                if sx != on or (sd != od and 0 not in (sd, od)):
                    continue
                s = mk(*ss)
                e = max(od, sd)
                X, N = LETTERS[:ox], LETTERS[ox:ox + on]
                Y, M = LETTERS[5:5 + sy], LETTERS[7:7 + sn]
                E = "z" * e
                expected = np.einsum(f"{X}{N}{E},{N}{Y}{M}{E}->{X}{Y}{M}{E}",
                                     add_dim(dense(other), od, e),
                                     add_dim(dense(s), sd, e))
                self.assertDense(s.dot_left(other), expected,
                                 ("dot_left", so, ss))

        # dot_right: other[X, Y, N, E] . self[N, M, E] -> [X, Y, M, E]
        for so in structures():
            ox, oy, on, od = so[:4]
            other = mk(*so)
            for ss in structures(max_x=2, max_y=0):
                sx, _, sn, sd = ss[:4]
                if sx != on or (sd != od and 0 not in (sd, od)):
                    continue
                s = mk(*ss)
                e = max(od, sd)
                X, Y = LETTERS[:ox], LETTERS[2:2 + oy]
                N, M = LETTERS[4:4 + on], LETTERS[7:7 + sn]
                E = "z" * e
                expected = np.einsum(f"{X}{Y}{N}{E},{N}{M}{E}->{X}{Y}{M}{E}",
                                     add_dim(dense(other), od, e),
                                     add_dim(dense(s), sd, e))
                self.assertDense(s.dot_right(other), expected,
                                 ("dot_right", so, ss))

    def test_maps_with_unmapped_entries(self):
        self.run_all(normalize=False)

    def test_normalized_maps(self):
        self.run_all(normalize=True)


if __name__ == "__main__":
    unittest.main()
