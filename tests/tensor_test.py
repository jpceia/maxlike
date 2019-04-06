import unittest
import numpy as np
from array import array
from maxlike.tensor import BaseTensor
from maxlike import Tensor


class Test(unittest.TestCase):

    def check_comm(self, foo, *t_args):
        m1 = foo(*t_args).toarray()
        arr_args = [t.toarray() if isinstance(t, BaseTensor)
                    else t for t in t_args]
        m2 = foo(*arr_args)
        msg = "\nM1=\n{}\n\nM2=\n{}\n"
        self.assertTrue(np.allclose(m1, m2), msg.format(m1, m2))

    def setUp(self):
        n = 3
        self.a = Tensor(np.arange(1, n * n + 1).reshape((n, n)))
        self.b = Tensor(np.arange(n * n + 1, 2 * n * n + 1).reshape((n, n)))
        self.a1 = Tensor(self.a.values[None, :, :], 1, 0, 0, array('b', [0]))
        self.b1 = Tensor(self.b.values[None, :, :], 1, 0, 0, array('b', [0]))
        self.b2 = Tensor(self.b.values[None, :, :], 1, 0, 0, array('b', [1]))
        self.c1 = Tensor(np.arange(1, n * n * n + 1).reshape((n, n, n)), 1, 0, 0)
        self.a12 = Tensor(self.a.values[None, None, :, :], 1, 1, 0,
                          array('b', [0]), array('b', [1]))
        self.b21 = Tensor(self.b.values[None, None, :, :], 1, 1, 0,
                          array('b', [0]), array('b', [1]))

    def test_bin_op(self):
        a = self.a
        b = self.b
        a1 = self.a1
        b1 = self.b1
        b2 = self.b2
        c1 = self.c1
        a12 = self.a12
        b21 = self.b21

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

        self.check_comm(lambda x, y: x + y, a, 0)
        self.check_comm(lambda x, y: x - y, a, 0)
        self.check_comm(lambda x, y: x * y, a, 0)

        self.check_comm(lambda x, y: x + y, a, 1)
        self.check_comm(lambda x, y: x - y, a, 1)
        self.check_comm(lambda x, y: x * y, a, 1)
        self.check_comm(lambda x, y: x / y, a, 1)

        self.check_comm(lambda x, y: x + y, 0, b)
        self.check_comm(lambda x, y: x - y, 0, b)
        self.check_comm(lambda x, y: x * y, 0, b)
        self.check_comm(lambda x, y: x / y, 0, b)

        self.check_comm(lambda x, y: x + y, 1, b)
        self.check_comm(lambda x, y: x - y, 1, b)
        self.check_comm(lambda x, y: x * y, 1, b)
        self.check_comm(lambda x, y: x / y, 1, b)

        self.check_comm(lambda x, y: x + y, a1, b)
        self.check_comm(lambda x, y: x - y, a1, b)
        self.check_comm(lambda x, y: x * y, a1, b)
        self.check_comm(lambda x, y: x / y, a1, b)

        self.check_comm(lambda x, y: x + y, a1, 0)
        self.check_comm(lambda x, y: x - y, a1, 0)
        self.check_comm(lambda x, y: x * y, a1, 0)

        self.check_comm(lambda x, y: x + y, a1, 1)
        self.check_comm(lambda x, y: x - y, a1, 1)
        self.check_comm(lambda x, y: x * y, a1, 1)
        self.check_comm(lambda x, y: x / y, a1, 1)

        self.check_comm(lambda x, y: x + y, a, b1)
        self.check_comm(lambda x, y: x - y, a, b1)
        self.check_comm(lambda x, y: x * y, a, b1)

        self.check_comm(lambda x, y: x + y, 0, b1)
        self.check_comm(lambda x, y: x - y, 0, b1)
        self.check_comm(lambda x, y: x * y, 0, b1)

        self.check_comm(lambda x, y: x + y, 1, b1)
        self.check_comm(lambda x, y: x - y, 1, b1)
        self.check_comm(lambda x, y: x * y, 1, b1)

        self.check_comm(lambda x, y: x + y, a, c1)
        self.check_comm(lambda x, y: x - y, a, c1)
        self.check_comm(lambda x, y: x * y, a, c1)

        self.check_comm(lambda x, y: x + y, 0, c1)
        self.check_comm(lambda x, y: x - y, 0, c1)
        self.check_comm(lambda x, y: x * y, 0, c1)

        self.check_comm(lambda x, y: x + y, 1, c1)
        self.check_comm(lambda x, y: x - y, 1, c1)
        self.check_comm(lambda x, y: x * y, 1, c1)

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

        self.check_comm(lambda x, y: x + y, a12, 0)
        self.check_comm(lambda x, y: x - y, a12, 0)
        self.check_comm(lambda x, y: x * y, a12, 0)

        self.check_comm(lambda x, y: x + y, a12, 1)
        self.check_comm(lambda x, y: x - y, a12, 1)
        self.check_comm(lambda x, y: x * y, a12, 1)
        self.check_comm(lambda x, y: x / y, a12, 1)

        self.check_comm(lambda x, y: x + y, a, b21)
        self.check_comm(lambda x, y: x - y, a, b21)
        self.check_comm(lambda x, y: x * y, a, b21)

        self.check_comm(lambda x, y: x + y, 0, b21)
        self.check_comm(lambda x, y: x - y, 0, b21)
        self.check_comm(lambda x, y: x * y, 0, b21)

        self.check_comm(lambda x, y: x + y, 1, b21)
        self.check_comm(lambda x, y: x - y, 1, b21)
        self.check_comm(lambda x, y: x * y, 1, b21)

        self.check_comm(lambda x, y: x + y, a12, a12)
        self.check_comm(lambda x, y: x - y, a12, a12)
        self.check_comm(lambda x, y: x * y, a12, a12)

        # Generic Tensor (left)
        self.check_comm(lambda x, y, z: (x + y) + z, a, b, a)
        self.check_comm(lambda x, y, z: (x + y) - z, a, b, a)
        self.check_comm(lambda x, y, z: (x + y) * z, a, b, a)
        self.check_comm(lambda x, y, z: (x + y) / z, a, b, a)

        self.check_comm(lambda x, y, z: (x + y) + z, a, b, 0)
        self.check_comm(lambda x, y, z: (x + y) - z, a, b, 0)
        self.check_comm(lambda x, y, z: (x + y) * z, a, b, 0)

        self.check_comm(lambda x, y, z: (x + y) + z, a, b, 1)
        self.check_comm(lambda x, y, z: (x + y) - z, a, b, 1)
        self.check_comm(lambda x, y, z: (x + y) * z, a, b, 1)
        self.check_comm(lambda x, y, z: (x + y) / z, a, b, 1)

        self.check_comm(lambda x, y, z: (x + y) + z, a1, b2, a)
        self.check_comm(lambda x, y, z: (x + y) - z, a1, b2, a)
        self.check_comm(lambda x, y, z: (x + y) * z, a1, b2, a)
        self.check_comm(lambda x, y, z: (x + y) / z, a1, b2, a)

        self.check_comm(lambda x, y, z: (x + y) + z, a1, b2, 0)
        self.check_comm(lambda x, y, z: (x + y) - z, a1, b2, 0)
        self.check_comm(lambda x, y, z: (x + y) * z, a1, b2, 0)

        self.check_comm(lambda x, y, z: (x + y) + z, a1, b2, 1)
        self.check_comm(lambda x, y, z: (x + y) - z, a1, b2, 1)
        self.check_comm(lambda x, y, z: (x + y) * z, a1, b2, 1)
        self.check_comm(lambda x, y, z: (x + y) / z, a1, b2, 1)

        self.check_comm(lambda x, y, z: (x + y) + z, a12, b21, a)
        self.check_comm(lambda x, y, z: (x + y) - z, a12, b21, a)
        self.check_comm(lambda x, y, z: (x + y) * z, a12, b21, a)
        self.check_comm(lambda x, y, z: (x + y) / z, a12, b21, a)

        self.check_comm(lambda x, y, z: (x + y) + z, a12, b21, 0)
        self.check_comm(lambda x, y, z: (x + y) - z, a12, b21, 0)
        self.check_comm(lambda x, y, z: (x + y) * z, a12, b21, 0)

        self.check_comm(lambda x, y, z: (x + y) + z, a12, b21, 1)
        self.check_comm(lambda x, y, z: (x + y) - z, a12, b21, 1)
        self.check_comm(lambda x, y, z: (x + y) * z, a12, b21, 1)
        self.check_comm(lambda x, y, z: (x + y) / z, a12, b21, 1)

        self.check_comm(lambda x, y, z: (x + y) + z, a, c1, b)
        self.check_comm(lambda x, y, z: (x + y) - z, a, c1, b)
        self.check_comm(lambda x, y, z: (x + y) * z, a, c1, b)
        self.check_comm(lambda x, y, z: (x + y) / z, a, c1, b)

        self.check_comm(lambda x, y, z: (x + y) + z, a, c1, 0)
        self.check_comm(lambda x, y, z: (x + y) - z, a, c1, 0)
        self.check_comm(lambda x, y, z: (x + y) * z, a, c1, 0)

        self.check_comm(lambda x, y, z: (x + y) + z, a, c1, 1)
        self.check_comm(lambda x, y, z: (x + y) - z, a, c1, 1)
        self.check_comm(lambda x, y, z: (x + y) * z, a, c1, 1)
        self.check_comm(lambda x, y, z: (x + y) / z, a, c1, 1)

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

        self.check_comm(lambda x, y, z: x + (y + z), 0, b, a)
        self.check_comm(lambda x, y, z: x - (y + z), 0, b, a)
        self.check_comm(lambda x, y, z: x * (y + z), 0, b, a)

        self.check_comm(lambda x, y, z: x + (y + z), 1, b, a)
        self.check_comm(lambda x, y, z: x - (y + z), 1, b, a)
        self.check_comm(lambda x, y, z: x * (y + z), 1, b, a)

        self.check_comm(lambda x, y, z: x + (y + z), a, b1, b2)
        self.check_comm(lambda x, y, z: x - (y + z), a, b1, b2)
        self.check_comm(lambda x, y, z: x * (y + z), a, b1, b2)

        self.check_comm(lambda x, y, z: x + (y + z), 0, b1, b2)
        self.check_comm(lambda x, y, z: x - (y + z), 0, b1, b2)
        self.check_comm(lambda x, y, z: x * (y + z), 0, b1, b2)

        self.check_comm(lambda x, y, z: x + (y + z), 1, b1, b2)
        self.check_comm(lambda x, y, z: x - (y + z), 1, b1, b2)
        self.check_comm(lambda x, y, z: x * (y + z), 1, b1, b2)

        self.check_comm(lambda x, y, z: x + (y + z), a1, b1, b2)
        self.check_comm(lambda x, y, z: x - (y + z), a1, b1, b2)
        self.check_comm(lambda x, y, z: x * (y + z), a1, b1, b2)

        self.check_comm(lambda x, y, z: x + (y + z), c1, b1, b2)
        self.check_comm(lambda x, y, z: x - (y + z), c1, b1, b2)
        self.check_comm(lambda x, y, z: x * (y + z), c1, b1, b2)

        self.check_comm(lambda x, y, z: x + (y + z), a, a12, b21)
        self.check_comm(lambda x, y, z: x - (y + z), a, a12, b21)
        self.check_comm(lambda x, y, z: x * (y + z), a, a12, b21)

        self.check_comm(lambda x, y, z: x + (y + z), 0, a12, b21)
        self.check_comm(lambda x, y, z: x - (y + z), 0, a12, b21)
        self.check_comm(lambda x, y, z: x * (y + z), 0, a12, b21)

        self.check_comm(lambda x, y, z: x + (y + z), 1, a12, b21)
        self.check_comm(lambda x, y, z: x - (y + z), 1, a12, b21)
        self.check_comm(lambda x, y, z: x * (y + z), 1, a12, b21)

        # checking that division by zero throws ZeroDivisionError
        for args in [
                (lambda x, y: x / y, a, 0),
                (lambda x, y: x / y, a1, 0),
                (lambda x, y: x / y, a12, 0),
                (lambda x, y, z: (x + y) / z, a, b, 0),
                (lambda x, y, z: (x + y) / z, a1, b2, 0),
                (lambda x, y, z: (x + y) / z, a12, b21, 0),
                (lambda x, y, z: (x + y) / z, a, c1, 0)
            ]:
            try:
                self.check_comm(*args)
            except ZeroDivisionError:
                pass

    def test_sum(self):
        a = self.a
        b = self.b
        a1 = self.a1
        b1 = self.b1
        b2 = self.b2
        c1 = self.c1
        a12 = self.a12
        b21 = self.b21

        tensor_sum = lambda x: x.sum() if isinstance(x, BaseTensor) \
                                       else x.sum((-1, -2)).transpose()
        self.check_comm(tensor_sum, a)
        self.check_comm(tensor_sum, a1)
        self.check_comm(tensor_sum, c1)
        self.check_comm(tensor_sum, a12)
        self.check_comm(tensor_sum, a1 + b2)
        self.check_comm(tensor_sum, a12 + b21)


if __name__ == "__main__":
    unittest.main()
