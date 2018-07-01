import unittest
import sys
sys.path.insert(0, "..")
from maxlike.tensor import *


class Test(unittest.TestCase):

    def check_comm(self, foo, *t_args):
        M1 = foo(*t_args).toarray()
        M2 = foo(*[t.toarray() for t in t_args])
        msg = "\nM1=\n{}\n\nM2=\n{}\n"
        self.assertTrue(np.allclose(M1, M2), msg.format(M1, M2))

    def test_tensor(self):
        n = 3
        a = np.arange(1, n * n + 1).reshape((n, n))
        b = np.arange(n * n + 1, 2 * n * n + 1).reshape((n, n))
        c = np.arange(1, n * n * n + 1).reshape((n, n, n))
        A, B = Tensor(a), Tensor(b)
        A1 = Tensor(a[None, :, :], 1, 0, 0, [0])
        A2 = Tensor(a[None, :, :], 1, 0, 0, [1])
        B1 = Tensor(b[None, :, :], 1, 0, 0, [0])
        B2 = Tensor(b[None, :, :], 1, 0, 0, [1])
        C1 = Tensor(c, 1, 0, 0,)
        A12 = Tensor(a[None, None, :, :], 1, 1, 0, [0], [1])
        B21 = Tensor(b[None, None, :, :], 1, 1, 0, [0], [1])

        self.check_comm(lambda x: -x, A)
        self.check_comm(lambda x: -x, A1)
        self.check_comm(lambda x: -x, A12)

        self.check_comm(lambda x, y: -(x + y), A, B)
        self.check_comm(lambda x, y: -(x + y), A, B1)
        self.check_comm(lambda x, y: -(x + y), A1, B2)

        self.check_comm(lambda x, y: -(x + y), A12, B)
        self.check_comm(lambda x, y: -(x + y), A12, B1)
        self.check_comm(lambda x, y: -(x + y), A12, B21)

        self.check_comm(lambda x, y: x + y, A, B)
        self.check_comm(lambda x, y: x - y, A, B)
        self.check_comm(lambda x, y: x * y, A, B)
        self.check_comm(lambda x, y: x / y, A, B)

        self.check_comm(lambda x, y: x + y, A1, B)
        self.check_comm(lambda x, y: x - y, A1, B)
        self.check_comm(lambda x, y: x * y, A1, B)
        self.check_comm(lambda x, y: x / y, A1, B)

        self.check_comm(lambda x, y: x + y, A, B1)
        self.check_comm(lambda x, y: x - y, A, B1)
        self.check_comm(lambda x, y: x * y, A, B1)

        self.check_comm(lambda x, y: x + y, A1, B1)
        self.check_comm(lambda x, y: x - y, A1, B1)
        self.check_comm(lambda x, y: x * y, A1, B1)

        self.check_comm(lambda x, y: x + y, A1, B2)
        self.check_comm(lambda x, y: x - y, A1, B2)
        self.check_comm(lambda x, y: x * y, A1, B2)

        self.check_comm(lambda x, y: x + y, A12, B)
        self.check_comm(lambda x, y: x - y, A12, B)
        self.check_comm(lambda x, y: x * y, A12, B)
        self.check_comm(lambda x, y: x / y, A12, B)

        self.check_comm(lambda x, y: x + y, A, B21)
        self.check_comm(lambda x, y: x - y, A, B21)
        self.check_comm(lambda x, y: x * y, A, B21)

        self.check_comm(lambda x, y: x + y, A12, B1)
        self.check_comm(lambda x, y: x - y, A12, B1)
        self.check_comm(lambda x, y: x * y, A12, B1)

        self.check_comm(lambda x, y: x + y, A1, B21)
        self.check_comm(lambda x, y: x - y, A1, B21)
        self.check_comm(lambda x, y: x * y, A1, B21)

        self.check_comm(lambda x, y: x + y, A12, A12)
        self.check_comm(lambda x, y: x - y, A12, A12)
        self.check_comm(lambda x, y: x * y, A12, A12)

        self.check_comm(lambda x, y: x + y, A12, B21)
        self.check_comm(lambda x, y: x - y, A12, B21)
        self.check_comm(lambda x, y: x * y, A12, B21)

        # Generic Tensor (left)
        self.check_comm(lambda x, y, z: (x + y) + z, A, B, A)
        self.check_comm(lambda x, y, z: (x + y) - z, A, B, A)
        self.check_comm(lambda x, y, z: (x + y) * z, A, B, A)
        self.check_comm(lambda x, y, z: (x + y) / z, A, B, A)

        self.check_comm(lambda x, y, z: (x + y) + z, A1, B2, A)
        self.check_comm(lambda x, y, z: (x + y) - z, A1, B2, A)
        self.check_comm(lambda x, y, z: (x + y) * z, A1, B2, A)
        self.check_comm(lambda x, y, z: (x + y) / z, A1, B2, A)

        self.check_comm(lambda x, y, z: (x + y) + z, A1, B2, A)
        self.check_comm(lambda x, y, z: (x + y) - z, A1, B2, A)
        self.check_comm(lambda x, y, z: (x + y) * z, A1, B2, A)
        self.check_comm(lambda x, y, z: (x + y) / z, A1, B2, A)

        self.check_comm(lambda x, y, z: (x + y) + z, A12, B21, A)
        self.check_comm(lambda x, y, z: (x + y) - z, A12, B21, A)
        self.check_comm(lambda x, y, z: (x + y) * z, A12, B21, A)
        self.check_comm(lambda x, y, z: (x + y) / z, A12, B21, A)

        self.check_comm(lambda x, y, z: x + (y + z), A, B1, B2)
        self.check_comm(lambda x, y, z: x - (y + z), A, B1, B2)
        self.check_comm(lambda x, y, z: x * (y + z), A, B1, B2)

        self.check_comm(lambda x, y, z: x + (y + z), A1, A12, B21)
        self.check_comm(lambda x, y, z: x - (y + z), A1, A12, B21)
        self.check_comm(lambda x, y, z: x * (y + z), A1, A12, B21)

        self.check_comm(lambda x, y, z: (x + y) + z, A12, B21, B1)
        self.check_comm(lambda x, y, z: (x + y) - z, A12, B21, B1)
        self.check_comm(lambda x, y, z: (x + y) * z, A12, B21, B1)

        self.check_comm(lambda x, y, z: (x + y) + z, A12, B21, B)
        self.check_comm(lambda x, y, z: (x + y) - z, A12, B21, B)
        self.check_comm(lambda x, y, z: (x + y) * z, A12, B21, B)
        self.check_comm(lambda x, y, z: (x + y) / z, A12, B21, B)

        self.check_comm(lambda x: -x, C1)
        self.check_comm(lambda x, y: -(x + y), A, C1)
        self.check_comm(lambda x, y: -(x + y), A1, C1)

        self.check_comm(lambda x, y: -(x + y), A12, B)
        self.check_comm(lambda x, y: -(x + y), A12, B1)
        self.check_comm(lambda x, y: -(x + y), A12, B21)

        self.check_comm(lambda x, y: x + y, A, B)
        self.check_comm(lambda x, y: x - y, A, B)
        self.check_comm(lambda x, y: x * y, A, B)
        self.check_comm(lambda x, y: x / y, A, B)

        self.check_comm(lambda x, y: x + y, A1, B)
        self.check_comm(lambda x, y: x - y, A1, B)
        self.check_comm(lambda x, y: x * y, A1, B)
        self.check_comm(lambda x, y: x / y, A1, B)

        self.check_comm(lambda x, y: x + y, A, B1)
        self.check_comm(lambda x, y: x - y, A, B1)
        self.check_comm(lambda x, y: x * y, A, B1)

        self.check_comm(lambda x, y: x + y, A, C1)
        self.check_comm(lambda x, y: x - y, A, C1)
        self.check_comm(lambda x, y: x * y, A, C1)

        self.check_comm(lambda x, y: x + y, A1, B1)
        self.check_comm(lambda x, y: x - y, A1, B1)
        self.check_comm(lambda x, y: x * y, A1, B1)

        self.check_comm(lambda x, y: x + y, A1, B2)
        self.check_comm(lambda x, y: x - y, A1, B2)
        self.check_comm(lambda x, y: x * y, A1, B2)

        self.check_comm(lambda x, y: x + y, A1, C1)
        self.check_comm(lambda x, y: x - y, A1, C1)
        self.check_comm(lambda x, y: x * y, A1, C1)

        self.check_comm(lambda x, y: x + y, A12, B)
        self.check_comm(lambda x, y: x - y, A12, B)
        self.check_comm(lambda x, y: x * y, A12, B)
        self.check_comm(lambda x, y: x / y, A12, B)

        self.check_comm(lambda x, y: x + y, A, B21)
        self.check_comm(lambda x, y: x - y, A, B21)
        self.check_comm(lambda x, y: x * y, A, B21)

        self.check_comm(lambda x, y: x + y, A12, B1)
        self.check_comm(lambda x, y: x - y, A12, B1)
        self.check_comm(lambda x, y: x * y, A12, B1)

        self.check_comm(lambda x, y: x + y, A1, B21)
        self.check_comm(lambda x, y: x - y, A1, B21)
        self.check_comm(lambda x, y: x * y, A1, B21)

        self.check_comm(lambda x, y: x + y, A12, A12)
        self.check_comm(lambda x, y: x - y, A12, A12)
        self.check_comm(lambda x, y: x * y, A12, A12)

        self.check_comm(lambda x, y: x + y, A12, B21)
        self.check_comm(lambda x, y: x - y, A12, B21)
        self.check_comm(lambda x, y: x * y, A12, B21)

        # Generic Tensor (left)
        self.check_comm(lambda x, y, z: (x + y) + z, A, C1, B)
        self.check_comm(lambda x, y, z: (x + y) - z, A, C1, B)
        self.check_comm(lambda x, y, z: (x + y) * z, A, C1, B)
        self.check_comm(lambda x, y, z: (x + y) / z, A, C1, B)

        self.check_comm(lambda x, y, z: (x + y) + z, A, B, C1)
        self.check_comm(lambda x, y, z: (x + y) - z, A, B, C1)
        self.check_comm(lambda x, y, z: (x + y) * z, A, B, C1)

        self.check_comm(lambda x, y, z: (x + y) + z, A1, B2, C1)
        self.check_comm(lambda x, y, z: (x + y) - z, A1, B2, C1)
        self.check_comm(lambda x, y, z: (x + y) * z, A1, B2, C1)

        self.check_comm(lambda x, y, z: (x + y) + z, A12, B21, C1)
        self.check_comm(lambda x, y, z: (x + y) - z, A12, B21, C1)
        self.check_comm(lambda x, y, z: (x + y) * z, A12, B21, C1)

        # Generic Tensor (right)
        self.check_comm(lambda x, y, z: x + (y + z), C1, B1, B2)
        self.check_comm(lambda x, y, z: x - (y + z), C1, B1, B2)
        self.check_comm(lambda x, y, z: x * (y + z), C1, B1, B2)

        self.check_comm(lambda x, y, z: x + (y + z), C1, A12, B21)
        self.check_comm(lambda x, y, z: x - (y + z), C1, A12, B21)
        self.check_comm(lambda x, y, z: x * (y + z), C1, A12, B21)


if __name__ == "__main__":
    unittest.main()
