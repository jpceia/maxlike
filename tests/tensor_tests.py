import unittest
import sys
sys.path.insert(0, "..")
from maxlike.tensor import *


class Test(unittest.TestCase):

    def test_same_mapping(self):
        A = np.arange(1, 5).reshape((2, 2))
        B = np.arange(5, 9).reshape((2, 2))
        TA1 = Tensor(A[None, :, :], 1, 0, 0, [0])
        TB1 = Tensor(B[None, :, :], 1, 0, 0, [0])
        aA1 = A[None, :, :] * np.eye(2)[:, :, None]
        aB1 = B[None, :, :] * np.eye(2)[:, :, None]

        M1 = (TA1 + TB1).sum().values
        M2 = (aA1 + aB1).sum((-1, -2)).flatten()
        np.testing.assert_array_equal(M1, M2)

        M1 = (TA1 - TB1).sum().values
        M2 = (aA1 - aB1).sum((-1, -2)).flatten()
        np.testing.assert_array_equal(M1, M2)

        M1 = (TA1 * TB1).sum().values
        M2 = (aA1 * aB1).sum((-1, -2)).flatten()
        np.testing.assert_array_equal(M1, M2)

        M1 = (TA1 / B).sum().values
        M2 = (aA1 / B).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)

    def test_different_mapping(self):
        A = np.arange(1, 5).reshape((2, 2))
        B = np.arange(5, 9).reshape((2, 2))
        TA1 = Tensor(A[None, :, :], 1, 0, 0, [0])
        TB2 = Tensor(B[None, :, :], 1, 0, 0, [1])
        aA1 = A[None, :, :] * np.eye(2)[:, :, None]
        aB2 = B[None, :, :] * np.eye(2)[:, None, :]

        M1 = (TA1 + TB2).sum().values
        M2 = (aA1 + aB2).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)

        M1 = (TA1 - TB2).sum().values
        M2 = (aA1 - aB2).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)

        M1 = (TA1 * TB2).sum().values
        M2 = (aA1 * aB2).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)

        M1 = (TA1 / B).sum().values
        M2 = (aA1 / B).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)

    @unittest.skip
    def test_generic_mapping(self):
        A = np.arange(1, 5).reshape((2, 2))
        B = np.arange(5, 9).reshape((2, 2))
        TA1 = Tensor(A[None, :, :], 1, 0, 0, [0])
        TB1 = Tensor(B[None, :, :], 1, 0, 0, [0])
        TB2 = Tensor(B[None, :, :], 1, 0, 0, [1])
        aA1 = A[None, :, :] * np.eye(2)[:, :, None]
        aB1 = B[None, :, :] * np.eye(2)[:, :, None]
        aB2 = B[None, :, :] * np.eye(2)[:, None, :]

        M1 = (TA1 + (TB1 + TB2)).sum().values
        M2 = (aA1 + (aB1 + aB2)).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)

        M1 = ((TB1 + TB2) + TA1).sum().values
        M2 = ((aB1 + aB2) + aA1).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)

        M1 = (TA1 - (TB1 + TB2)).sum().values
        M2 = (aA1 - (aB1 + aB2)).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)

        M1 = (TA1 * (TB1 + TB2)).sum().values
        M2 = (aA1 * (aB1 + aB2)).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)

        M1 = ((TB1 + TB2) / A).sum().values
        M2 = ((aB1 + aB2)  / A).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)

    @unittest.skip
    def test_hess(self):
        A = np.arange(1, 5).reshape((2, 2))
        B = np.arange(5, 9).reshape((2, 2))
        TA12 = Tensor(A[None, None, :, :], 1, 1, 0, [0], [0])
        TB1 = Tensor(B[None, :, :], 1, 0, 0, [0])
        TB2 = Tensor(B[None, :, :], 1, 0, 0, [1])
        TB12 = Tensor(B[None, None, :, :], 1, 1, 0, [0], [1])
        aA12 = A[None, None, :, :] * np.eye(2)[:, None, :, None] * np.eye(2)[None, :, None, :]
        aB1 = B[None, :, :] * np.eye(2)[:, :, None]
        aB2 = B[None, :, :] * np.eye(2)[:, None, :]

        M1 = (TB1 + TB2 + TA12).sum().values
        M2 = (aB1 + aB2 + aA12).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)

        M1 = (TB1 + TB2 - TA12).sum().values
        M2 = (aB1 + aB2 - aA12).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)

        M1 = (TB1 * TA12).sum().values
        M2 = (aB1 * aA12).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)

        M1 = (TB12 / A).sum().values
        M2 = (aB12 / A).sum((-1, -2))
        np.testing.assert_array_equal(M1, M2)


if __name__ == "__main__":
    unittest.main()