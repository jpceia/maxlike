import unittest
import numpy as np
# import sys; sys.path.insert(0 , "..")
from maxlike import Tensor


class Test(unittest.TestCase):

    def check_comm(self, foo, *t_args):
        m1 = foo(*t_args).toarray()
        m2 = foo(*[t.toarray() for t in t_args])
        msg = "\nM1=\n{}\n\nM2=\n{}\n"
        self.assertTrue(np.allclose(m1, m2), msg.format(m1, m2))

    def test_bin_op(self):
        n = 3
        a = Tensor(np.arange(1, n * n + 1).reshape((n, n)))
        b = Tensor(np.arange(n * n + 1, 2 * n * n + 1).reshape((n, n)))
        a1 = Tensor(a.values[None, :, :], 1, 0, 0, [0])
        b1 = Tensor(b.values[None, :, :], 1, 0, 0, [0])
        b2 = Tensor(b.values[None, :, :], 1, 0, 0, [1])
        c1 = Tensor(np.arange(1, n * n * n + 1).reshape((n, n, n)), 1, 0, 0,)
        a12 = Tensor(a.values[None, None, :, :], 1, 1, 0, [0], [1])
        b21 = Tensor(b.values[None, None, :, :], 1, 1, 0, [0], [1])

        self.check_comm(lambda x: -x, a)
        self.check_comm(lambda x: -x, a1)
        self.check_comm(lambda x: -x, a12)

        self.check_comm(lambda x, y: -(x + y), a, b)
        self.check_comm(lambda x, y: -(x + y), a1, b2)

        self.check_comm(lambda x, y: -(x + y), a12, b)
        self.check_comm(lambda x, y: -(x + y), a12, b21)

        self.check_comm(lambda x: -x, c1)
        self.check_comm(lambda x, y: -(x + y), a, c1)
        self.check_comm(lambda x, y: -(x + y), a1, c1)

        self.check_comm(lambda x, y: -(x + y), a12, b)
        self.check_comm(lambda x, y: -(x + y), a12, b21)

        self.check_comm(lambda x, y: x + y, a, b)
        self.check_comm(lambda x, y: x - y, a, b)
        self.check_comm(lambda x, y: x * y, a, b)
        self.check_comm(lambda x, y: x / y, a, b)

        self.check_comm(lambda x, y: x + y, a1, b)
        self.check_comm(lambda x, y: x - y, a1, b)
        self.check_comm(lambda x, y: x * y, a1, b)
        self.check_comm(lambda x, y: x / y, a1, b)

        self.check_comm(lambda x, y: x + y, a, b1)
        self.check_comm(lambda x, y: x - y, a, b1)
        self.check_comm(lambda x, y: x * y, a, b1)

        self.check_comm(lambda x, y: x + y, a, c1)
        self.check_comm(lambda x, y: x - y, a, c1)
        self.check_comm(lambda x, y: x * y, a, c1)

        self.check_comm(lambda x, y: x + y, a1, b1)
        self.check_comm(lambda x, y: x - y, a1, b1)
        self.check_comm(lambda x, y: x * y, a1, b1)

        self.check_comm(lambda x, y: x + y, a1, b2)
        self.check_comm(lambda x, y: x - y, a1, b2)
        self.check_comm(lambda x, y: x * y, a1, b2)

        self.check_comm(lambda x, y: x + y, a1, c1)
        self.check_comm(lambda x, y: x - y, a1, c1)
        self.check_comm(lambda x, y: x * y, a1, c1)

        self.check_comm(lambda x, y: x + y, a12, b)
        self.check_comm(lambda x, y: x - y, a12, b)
        self.check_comm(lambda x, y: x * y, a12, b)
        self.check_comm(lambda x, y: x / y, a12, b)

        self.check_comm(lambda x, y: x + y, a, b21)
        self.check_comm(lambda x, y: x - y, a, b21)
        self.check_comm(lambda x, y: x * y, a, b21)

        self.check_comm(lambda x, y: x + y, a12, a12)
        self.check_comm(lambda x, y: x - y, a12, a12)
        self.check_comm(lambda x, y: x * y, a12, a12)

        # Generic Tensor (left)
        self.check_comm(lambda x, y, z: (x + y) + z, a, b, a)
        self.check_comm(lambda x, y, z: (x + y) - z, a, b, a)
        self.check_comm(lambda x, y, z: (x + y) * z, a, b, a)
        self.check_comm(lambda x, y, z: (x + y) / z, a, b, a)

        self.check_comm(lambda x, y, z: (x + y) + z, a1, b2, a)
        self.check_comm(lambda x, y, z: (x + y) - z, a1, b2, a)
        self.check_comm(lambda x, y, z: (x + y) * z, a1, b2, a)
        self.check_comm(lambda x, y, z: (x + y) / z, a1, b2, a)

        self.check_comm(lambda x, y, z: (x + y) + z, a12, b21, a)
        self.check_comm(lambda x, y, z: (x + y) - z, a12, b21, a)
        self.check_comm(lambda x, y, z: (x + y) * z, a12, b21, a)
        self.check_comm(lambda x, y, z: (x + y) / z, a12, b21, a)

        self.check_comm(lambda x, y, z: (x + y) + z, a, c1, b)
        self.check_comm(lambda x, y, z: (x + y) - z, a, c1, b)
        self.check_comm(lambda x, y, z: (x + y) * z, a, c1, b)
        self.check_comm(lambda x, y, z: (x + y) / z, a, c1, b)

        self.check_comm(lambda x, y, z: (x + y) + z, a, b, c1)
        self.check_comm(lambda x, y, z: (x + y) - z, a, b, c1)
        self.check_comm(lambda x, y, z: (x + y) * z, a, b, c1)

        self.check_comm(lambda x, y, z: (x + y) + z, a1, b2, c1)
        self.check_comm(lambda x, y, z: (x + y) - z, a1, b2, c1)
        self.check_comm(lambda x, y, z: (x + y) * z, a1, b2, c1)

        # Generic Tensor (right)
        self.check_comm(lambda x, y, z: x + (y + z), a, b, a)
        self.check_comm(lambda x, y, z: x - (y + z), a, b, a)
        self.check_comm(lambda x, y, z: x * (y + z), a, b, a)

        self.check_comm(lambda x, y, z: x + (y + z), a, b1, b2)
        self.check_comm(lambda x, y, z: x - (y + z), a, b1, b2)
        self.check_comm(lambda x, y, z: x * (y + z), a, b1, b2)

        self.check_comm(lambda x, y, z: x + (y + z), a1, b1, b2)
        self.check_comm(lambda x, y, z: x - (y + z), a1, b1, b2)
        self.check_comm(lambda x, y, z: x * (y + z), a1, b1, b2)

        self.check_comm(lambda x, y, z: x + (y + z), c1, b1, b2)
        self.check_comm(lambda x, y, z: x - (y + z), c1, b1, b2)
        self.check_comm(lambda x, y, z: x * (y + z), c1, b1, b2)

        self.check_comm(lambda x, y, z: x + (y + z), a, a12, b21)
        self.check_comm(lambda x, y, z: x - (y + z), a, a12, b21)
        self.check_comm(lambda x, y, z: x * (y + z), a, a12, b21)


if __name__ == "__main__":
    unittest.main()
