"""
Algebra of the observation-indexed derivatives (IndexedTensor, SparseBlock):
every operation is compared with the same operation on the equivalent dense
Tensor.
"""
import unittest
import numpy as np
import scipy.sparse as sp
from array import array
from maxlike.tensor import Tensor, IndexedTensor, SparseBlock
from maxlike.tensor.tensor import GenericTensor, InvalidOperation
from maxlike.func import Take, X, Exp


rng = np.random.RandomState(0)
N_OBS, P_SIZE, Q_SIZE, E_SIZE = 7, 4, 3, 5
IDX_P = rng.randint(0, P_SIZE, N_OBS)
IDX_P2 = rng.randint(0, P_SIZE, N_OBS)
IDX_Q = rng.randint(0, Q_SIZE, N_OBS)


def grad(idx, size, weight=None):
    return Take(idx, weight).grad([np.zeros(size)], 0)


def dense(t):
    """Equivalent dense Tensor"""
    if isinstance(t, (IndexedTensor, SparseBlock)):
        return Tensor(t.toarray(), x_dim=t.x_dim, y_dim=t.y_dim, dim=t.dim)
    return t


class Test(unittest.TestCase):

    def same(self, a, b):
        a = a.toarray() if hasattr(a, "toarray") else np.asarray(a)
        b = b.toarray() if hasattr(b, "toarray") else np.asarray(b)
        np.testing.assert_allclose(*np.broadcast_arrays(a, b), atol=1e-12)

    def setUp(self):
        self.dp = grad(IDX_P, P_SIZE, rng.normal(size=N_OBS))
        self.dp2 = grad(IDX_P2, P_SIZE)
        self.dq = grad(IDX_Q, Q_SIZE, rng.normal(size=N_OBS))
        self.obs = Tensor(rng.normal(size=N_OBS))
        self.obs_e = Tensor(rng.uniform(1, 2, (N_OBS, E_SIZE)), dim=1)

    def test_toarray(self):
        arr = self.dp2.toarray()
        self.assertEqual(arr.shape, (P_SIZE, N_OBS))
        expected = np.zeros((P_SIZE, N_OBS))
        expected[IDX_P2, np.arange(N_OBS)] = 1
        np.testing.assert_array_equal(arr, expected)

    def test_scalar_ops(self):
        d = self.dp
        for res, ref in [(-d, -dense(d)), (2. * d, 2. * dense(d)),
                         (d * 2., dense(d) * 2.), (d / 4., dense(d) / 4.),
                         (np.float64(3.) * d, 3. * dense(d)),
                         (d + 0, dense(d)), (0 - d, -dense(d)),
                         (Tensor(0) + d, dense(d)),
                         (Tensor(0) - d, -dense(d))]:
            self.assertIsInstance(res, IndexedTensor)
            self.same(res, ref)
        self.same(d * 0, Tensor(0))
        self.same(Tensor(0) * d, Tensor(0))
        with self.assertRaises(InvalidOperation):
            d + 1.

    def test_values_product(self):
        d = self.dp
        for values in (self.obs, self.obs_e):
            self.same(d * values, dense(d) * values)
            self.same(values * d, values * dense(d))
            self.same(d / values, dense(d) / values)
        self.same(self.obs.values * d, self.obs * dense(d))
        res = d * self.obs_e
        self.assertEqual((res.x_dim, res.y_dim, res.n, res.dim),
                         (1, 0, 1, 1))

    def test_add_same_parameter(self):
        res = self.dp - self.dp2
        self.assertEqual(len(res.terms), 2)
        self.same(res, dense(self.dp) - dense(self.dp2))
        # terms with the same index are merged
        self.assertEqual(len((self.dp + 2. * self.dp).terms), 1)
        self.same(self.dp + 2. * self.dp, 3. * dense(self.dp))
        with self.assertRaises(InvalidOperation):
            self.dp + self.dq

    def test_transpose_and_pairs(self):
        t = self.dq.transpose()
        self.assertEqual((t.x_dim, t.y_dim), (0, 1))
        self.same(t, dense(self.dq).transpose())
        pair = self.dp * t
        self.assertEqual((pair.x_shape, pair.y_shape),
                         ((P_SIZE, ), (Q_SIZE, )))
        self.same(pair, dense(self.dp) * dense(self.dq).transpose())
        self.same(pair.transpose(),
                  dense(self.dq) * dense(self.dp).transpose())
        self.same(pair * self.obs_e,
                  dense(self.dp) * dense(self.dq).transpose() * self.obs_e)
        with self.assertRaises(InvalidOperation):
            self.dp * self.dp2

    def test_sum_gradient(self):
        d = self.dp * self.obs
        res = d.sum()
        self.assertIsInstance(res, Tensor)
        self.assertEqual(res.values.shape, (P_SIZE, ))
        self.same(res, dense(d).sum())
        d = self.dp * self.obs_e
        self.same(d.sum(), dense(d).sum())
        self.same(d.sum(False), dense(d).sum(False))
        self.assertIsInstance(d.sum(False), IndexedTensor)
        self.same(d.sum(True, False), dense(d).sum(True, False))
        self.same(d.transpose().sum(), dense(d).transpose().sum())

    def test_sum_hessian(self):
        a = self.dp - self.dp2
        for h in (a * self.dq.transpose() * self.obs,
                  a * a.transpose() * self.obs_e):
            res = h.sum()
            self.assertIsInstance(res, SparseBlock)
            self.assertTrue(sp.issparse(res.matrix))
            self.same(res, dense(h).sum())
            self.same(h.sum(False), dense(h).sum(False))
            self.same(h.sum(True, False), dense(h).sum(True, False))
        # one parameter entry per observation on both sides: a diagonal
        res = (self.dp * self.dp.transpose()).sum()
        self.assertLessEqual(res.matrix.nnz, P_SIZE)
        self.same(res, np.diag(np.diag(res.toarray())))

    def test_broadcast_sum(self):
        # values that broadcast along the distribution axis count once per
        # entry of that axis
        d = self.dp * self.obs_e - self.dp2
        self.same(d.sum(), dense(d).sum())
        self.same(d.sum(False), dense(d).sum(False))

    def test_sparse_block(self):
        h = (self.dp * self.dq.transpose() * self.obs).sum()
        ref = dense(h)
        self.same(h.values, ref)
        self.same(-h, -ref)
        self.same(2. * h, 2. * ref)
        self.same(h / 2., ref / 2.)
        self.same(h + h, 2. * ref)
        self.same(h - 3. * h, -2. * ref)
        self.same(h + Tensor(0), ref)
        self.same(Tensor(0) - h, -ref)
        self.same(h.transpose(), ref.transpose())
        self.assertIsInstance(h + 0, SparseBlock)
        full = Tensor(rng.normal(size=(P_SIZE, Q_SIZE)), x_dim=1, y_dim=1)
        for res, expected in [(h + full, ref + full), (full - h, full - ref)]:
            self.assertIsInstance(res, Tensor)
            self.same(res, expected)

    def test_structure(self):
        d = self.dp * self.obs_e
        ref = dense(d)
        self.same(d.expand([1], 2), ref.expand([1], 2))
        self.same(d.expand([0], 2, dim=True), ref.expand([0], 2, dim=True))
        self.same(d.flip([0], dim=True), ref.flip([0], dim=True))
        self.same(d.flip(0).sum(), ref.sum())
        dropped = d.drop_dim()
        self.assertEqual((dropped.n, dropped.dim), (2, 0))
        self.same(dropped, ref.drop_dim())
        self.assertIs(d.expand([0], 1), d)
        self.assertIs(d.flip(None), d)

    def test_from_tensor(self):
        values = rng.normal(size=(1, Q_SIZE, N_OBS, E_SIZE))
        for t in [
                # diagonal x, dense y
                Tensor(values, x_dim=1, y_dim=1, dim=1,
                       x_map=array('b', [0])),
                # x diagonal along the distribution axis
                Tensor(values, x_dim=1, y_dim=1, dim=1,
                       x_map=array('b', [1])),
                # dense x and y
                Tensor(values, x_dim=1, y_dim=1, dim=1),
                X().grad([np.zeros(N_OBS)], 0)]:
            self.same(IndexedTensor.from_tensor(t), t)

    def test_mixed_with_tensors(self):
        # one parameter per observation (diagonal Tensor) and a dense one
        diag = X().grad([np.zeros(N_OBS)], 0)
        full = Tensor(rng.normal(size=(Q_SIZE, N_OBS)), y_dim=1)
        self.same(self.dp * diag.transpose(),
                  dense(self.dp) * diag.transpose())
        self.same(self.dp * full, dense(self.dp) * full)
        self.same(full * self.dp, dense(self.dp) * full)
        res = (self.dp * full).sum()
        self.assertIsInstance(res, SparseBlock)
        self.same(res, (dense(self.dp) * full).sum())
        generic = GenericTensor(1, 0, 1, 0, [dense(self.dp2)])
        self.same(self.dp + generic, dense(self.dp) + dense(self.dp2))
        self.same(generic - self.dp, dense(self.dp2) - dense(self.dp))

    def test_chain_rule(self):
        # element-wise outer function: the index structure is kept
        y = Tensor(rng.normal(size=N_OBS))
        _, (f_grad, ), ((f_hess, ), ) = Exp().eval([y])
        d = self.dp - self.dp2
        res = f_grad.dot_left(d)
        self.assertIsInstance(res, IndexedTensor)
        self.same(res, f_grad.dot_left(dense(d)))
        ref = f_hess.dot_left(dense(d)).transpose().dot_left(
            dense(self.dq)).transpose()
        res = f_hess.dot_left(d).transpose().dot_left(self.dq).transpose()
        self.assertIsInstance(res, IndexedTensor)
        self.assertEqual(len(res.terms), 2)
        self.same(res, ref)
        h = d * self.dq.transpose() * self.obs
        self.same(f_grad.dot_right(h), f_grad.dot_right(dense(h)))
        # derivative with respect to a scalar parameter on one side
        self.same(f_hess.dot_left(d).transpose().dot_left(self.obs),
                  f_hess.dot_left(dense(d)).transpose().dot_left(self.obs))
        # outer function that is not element-wise: dense fallback
        full = Tensor(rng.normal(size=(N_OBS, 2)), x_dim=1)
        res = full.dot_left(d)
        self.assertIsInstance(res, Tensor)
        self.same(res, full.dot_left(dense(d)))

    def test_take_errors(self):
        with self.assertRaises(IndexError):
            Take([0, 4])([np.zeros(4)])
        with self.assertRaises(ValueError):
            Take([0, 1])([np.zeros((2, 2))])
        with self.assertRaises(TypeError):
            Take([0., 1.])


if __name__ == "__main__":
    unittest.main()
