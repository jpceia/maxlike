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

    def check_comm2(self, foo1, foo2, *t_args):
        m1 = foo1(*t_args).toarray()
        arr_args = [t.toarray() if isinstance(t, BaseTensor)
                    else t for t in t_args]
        m2 = foo2(*arr_args)
        msg = "\nM1=\n{}\n\nM2=\n{}\n"
        self.assertTrue(np.allclose(m1, m2), msg.format(m1, m2))

    def setUp(self):
        n = 3
        self.a = Tensor(np.arange(1, n * n + 1).reshape((n, n)))
        self.b = Tensor(np.arange(n * n + 1, 2 * n * n + 1).reshape((n, n)))
        self.c = Tensor(np.arange(n * n + 1, 3 * n * n + 1).reshape((n, n, 2)), dim=1)
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
        c = self.c
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

        self.check_comm(lambda x, y: x + y, a, 2)
        self.check_comm(lambda x, y: x - y, a, 2)
        self.check_comm(lambda x, y: x * y, a, 2)
        self.check_comm(lambda x, y: x / y, a, 2)

        self.check_comm(lambda x, y: x + y, a, b.values)
        self.check_comm(lambda x, y: x - y, a, b.values)
        self.check_comm(lambda x, y: x * y, a, b.values)
        self.check_comm(lambda x, y: x / y, a, b.values)

        self.check_comm(lambda x, y: x + y, 0, b)
        self.check_comm(lambda x, y: x - y, 0, b)
        self.check_comm(lambda x, y: x * y, 0, b)
        self.check_comm(lambda x, y: x / y, 0, b)

        self.check_comm(lambda x, y: x + y, 1, b)
        self.check_comm(lambda x, y: x - y, 1, b)
        self.check_comm(lambda x, y: x * y, 1, b)
        self.check_comm(lambda x, y: x / y, 1, b)

        self.check_comm(lambda x, y: x + y, 2, b)
        self.check_comm(lambda x, y: x - y, 2, b)
        self.check_comm(lambda x, y: x * y, 2, b)
        self.check_comm(lambda x, y: x / y, 2, b)

        self.check_comm(lambda x, y: x + y, a.values, b)
        self.check_comm(lambda x, y: x - y, a.values, b)
        self.check_comm(lambda x, y: x * y, a.values, b)
        self.check_comm(lambda x, y: x / y, a.values, b)

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

        self.check_comm(lambda x, y: x + y, a1, 2)
        self.check_comm(lambda x, y: x - y, a1, 2)
        self.check_comm(lambda x, y: x * y, a1, 2)
        self.check_comm(lambda x, y: x / y, a1, 2)

        self.check_comm(lambda x, y: x + y, a1, b.values)
        self.check_comm(lambda x, y: x - y, a1, b.values)
        self.check_comm(lambda x, y: x * y, a1, b.values)
        self.check_comm(lambda x, y: x / y, a1, b.values)

        self.check_comm(lambda x, y: x + y, a, b1)
        self.check_comm(lambda x, y: x - y, a, b1)
        self.check_comm(lambda x, y: x * y, a, b1)

        self.check_comm(lambda x, y: x + y, 0, b1)
        self.check_comm(lambda x, y: x - y, 0, b1)
        self.check_comm(lambda x, y: x * y, 0, b1)

        self.check_comm(lambda x, y: x + y, 1, b1)
        self.check_comm(lambda x, y: x - y, 1, b1)
        self.check_comm(lambda x, y: x * y, 1, b1)

        self.check_comm(lambda x, y: x + y, 2, b1)
        self.check_comm(lambda x, y: x - y, 2, b1)
        self.check_comm(lambda x, y: x * y, 2, b1)

        self.check_comm(lambda x, y: x + y, a.values, b1)
        self.check_comm(lambda x, y: x - y, a.values, b1)
        self.check_comm(lambda x, y: x * y, a.values, b1)

        self.check_comm(lambda x, y: x + y, a, c1)
        self.check_comm(lambda x, y: x - y, a, c1)
        self.check_comm(lambda x, y: x * y, a, c1)

        self.check_comm(lambda x, y: x + y, 0, c1)
        self.check_comm(lambda x, y: x - y, 0, c1)
        self.check_comm(lambda x, y: x * y, 0, c1)

        self.check_comm(lambda x, y: x + y, 1, c1)
        self.check_comm(lambda x, y: x - y, 1, c1)
        self.check_comm(lambda x, y: x * y, 1, c1)

        self.check_comm(lambda x, y: x + y, 2, c1)
        self.check_comm(lambda x, y: x - y, 2, c1)
        self.check_comm(lambda x, y: x * y, 2, c1)

        self.check_comm(lambda x, y: x + y, a.values, c1)
        self.check_comm(lambda x, y: x - y, a.values, c1)
        self.check_comm(lambda x, y: x * y, a.values, c1)

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

        self.check_comm(lambda x, y: x + y, a12, 2)
        self.check_comm(lambda x, y: x - y, a12, 2)
        self.check_comm(lambda x, y: x * y, a12, 2)
        self.check_comm(lambda x, y: x / y, a12, 2)

        self.check_comm(lambda x, y: x + y, a12, a.values)
        self.check_comm(lambda x, y: x - y, a12, a.values)
        self.check_comm(lambda x, y: x * y, a12, a.values)
        self.check_comm(lambda x, y: x / y, a12, a.values)

        self.check_comm(lambda x, y: x + y, a, b21)
        self.check_comm(lambda x, y: x - y, a, b21)
        self.check_comm(lambda x, y: x * y, a, b21)

        self.check_comm(lambda x, y: x + y, 0, b21)
        self.check_comm(lambda x, y: x - y, 0, b21)
        self.check_comm(lambda x, y: x * y, 0, b21)

        self.check_comm(lambda x, y: x + y, 1, b21)
        self.check_comm(lambda x, y: x - y, 1, b21)
        self.check_comm(lambda x, y: x * y, 1, b21)

        self.check_comm(lambda x, y: x + y, 2, b21)
        self.check_comm(lambda x, y: x - y, 2, b21)
        self.check_comm(lambda x, y: x * y, 2, b21)

        self.check_comm(lambda x, y: x + y, a.values, b21)
        self.check_comm(lambda x, y: x - y, a.values, b21)
        self.check_comm(lambda x, y: x * y, a.values, b21)

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

        self.check_comm(lambda x, y, z: (x + y) + z, a, b, 2)
        self.check_comm(lambda x, y, z: (x + y) - z, a, b, 2)
        self.check_comm(lambda x, y, z: (x + y) * z, a, b, 2)
        self.check_comm(lambda x, y, z: (x + y) / z, a, b, 2)

        self.check_comm(lambda x, y, z: (x + y) + z, a, b, a.values)
        self.check_comm(lambda x, y, z: (x + y) - z, a, b, a.values)
        self.check_comm(lambda x, y, z: (x + y) * z, a, b, a.values)
        self.check_comm(lambda x, y, z: (x + y) / z, a, b, a.values)

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

        self.check_comm(lambda x, y, z: (x + y) + z, a1, b2, 2)
        self.check_comm(lambda x, y, z: (x + y) - z, a1, b2, 2)
        self.check_comm(lambda x, y, z: (x + y) * z, a1, b2, 2)
        self.check_comm(lambda x, y, z: (x + y) / z, a1, b2, 2)

        self.check_comm(lambda x, y, z: (x + y) + z, a1, b2, a.values)
        self.check_comm(lambda x, y, z: (x + y) - z, a1, b2, a.values)
        self.check_comm(lambda x, y, z: (x + y) * z, a1, b2, a.values)
        self.check_comm(lambda x, y, z: (x + y) / z, a1, b2, a.values)

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

        self.check_comm(lambda x, y, z: (x + y) + z, a12, b21, 2)
        self.check_comm(lambda x, y, z: (x + y) - z, a12, b21, 2)
        self.check_comm(lambda x, y, z: (x + y) * z, a12, b21, 2)
        self.check_comm(lambda x, y, z: (x + y) / z, a12, b21, 2)

        self.check_comm(lambda x, y, z: (x + y) + z, a12, b21, a.values)
        self.check_comm(lambda x, y, z: (x + y) - z, a12, b21, a.values)
        self.check_comm(lambda x, y, z: (x + y) * z, a12, b21, a.values)
        self.check_comm(lambda x, y, z: (x + y) / z, a12, b21, a.values)

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

        self.check_comm(lambda x, y, z: (x + y) + z, a, c1, 2)
        self.check_comm(lambda x, y, z: (x + y) - z, a, c1, 2)
        self.check_comm(lambda x, y, z: (x + y) * z, a, c1, 2)
        self.check_comm(lambda x, y, z: (x + y) / z, a, c1, 2)

        self.check_comm(lambda x, y, z: (x + y) + z, a, c1, a.values)
        self.check_comm(lambda x, y, z: (x + y) - z, a, c1, a.values)
        self.check_comm(lambda x, y, z: (x + y) * z, a, c1, a.values)
        self.check_comm(lambda x, y, z: (x + y) / z, a, c1, a.values)

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

        self.check_comm(lambda x, y, z: x + (y + z), 2, b, a)
        self.check_comm(lambda x, y, z: x - (y + z), 2, b, a)
        self.check_comm(lambda x, y, z: x * (y + z), 2, b, a)

        self.check_comm(lambda x, y, z: x + (y + z), a.values, b, a)
        self.check_comm(lambda x, y, z: x - (y + z), a.values, b, a)
        self.check_comm(lambda x, y, z: x * (y + z), a.values, b, a)

        self.check_comm(lambda x, y, z: x + (y + z), a, b1, b2)
        self.check_comm(lambda x, y, z: x - (y + z), a, b1, b2)
        self.check_comm(lambda x, y, z: x * (y + z), a, b1, b2)

        self.check_comm(lambda x, y, z: x + (y + z), 0, b1, b2)
        self.check_comm(lambda x, y, z: x - (y + z), 0, b1, b2)
        self.check_comm(lambda x, y, z: x * (y + z), 0, b1, b2)

        self.check_comm(lambda x, y, z: x + (y + z), 1, b1, b2)
        self.check_comm(lambda x, y, z: x - (y + z), 1, b1, b2)
        self.check_comm(lambda x, y, z: x * (y + z), 1, b1, b2)

        self.check_comm(lambda x, y, z: x + (y + z), 2, b1, b2)
        self.check_comm(lambda x, y, z: x - (y + z), 2, b1, b2)
        self.check_comm(lambda x, y, z: x * (y + z), 2, b1, b2)

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

        self.check_comm(lambda x, y, z: x + (y + z), 2, a12, b21)
        self.check_comm(lambda x, y, z: x - (y + z), 2, a12, b21)
        self.check_comm(lambda x, y, z: x * (y + z), 2, a12, b21)

        self.check_comm(lambda x, y, z: x + (y + z), a.values, a12, b21)
        self.check_comm(lambda x, y, z: x - (y + z), a.values, a12, b21)
        self.check_comm(lambda x, y, z: x * (y + z), a.values, a12, b21)

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
        sum_tensor = lambda x: x.sum()
        sum_array = lambda x: x.sum((-1, -2))
        check = lambda x: self.check_comm2(sum_tensor, sum_array, x)

        check(self.a)
        check(self.a1)
        check(self.c1)
        check(self.a12)
        check(self.a1 + self.b1)
        check(self.a12 + self.b21)

        sum_tensor = lambda x: x.sum(sum_val=False)
        sum_array = lambda x: x
        check = lambda x: self.check_comm2(sum_tensor, sum_array, x)

        check(self.a)
        check(self.a1)
        check(self.c1)
        check(self.a12)
        check(self.a1 + self.b1)
        check(self.a12 + self.b21)

        self.check_comm2(
            lambda x: x.sum(sum_val=True, sum_dim=True),
            lambda x: x.sum(), self.c)

        self.check_comm2(
            lambda x: x.sum(sum_val=False, sum_dim=False),
            lambda x: x, self.c)

        self.check_comm2(
            lambda x: x.sum(sum_val=False, sum_dim=True),
            lambda x: x.sum(-1), self.c)

    def test_flip(self):
        flip_tensor = lambda x: x.flip([0, 1])
        flip_array = lambda x: np.flip(x, [-1, -2])
        check = lambda x: self.check_comm2(flip_tensor, flip_array, x)

        check(self.a)
        check(self.c1)

        flip_tensor = lambda x: x.flip(0, dim=True)
        flip_array = lambda x: np.flip(x, -1)
        self.check_comm2(flip_tensor, flip_array, self.c)

    def test_dot_left(self):

        def array_dot_left(left, right):
            assert isinstance(left, BaseTensor)
            assert isinstance(right, BaseTensor)
            assert right.y_dim == 0
            assert right.n == left.x_dim
            dim = max(right.dim, left.dim)
            r_idx = [slice(None)] * (right.x_dim + right.n) + \
                    [None] * (left.y_dim + left.n) + \
                    [slice(None)] * right.dim + \
                    [None] * (dim - right.dim)
            l_idx = [None] * right.x_dim + [...] + \
                    [None] * (dim - left.dim)
            val = np.multiply(
                left.toarray()[tuple(l_idx)],
                right.toarray()[tuple(r_idx)])
            val = val.sum(tuple(right.x_dim + np.arange(right.n)))
            return val

        n = 3
        msg = "\nM1=\n{}\n\nM2=\n{}\n"
        d = Tensor(np.arange(2, n + 2)[None, :], 1, 0, 0, array('b', [0]))

        m1 = d.dot(d).toarray()
        m2 = array_dot_left(d, d)
        self.assertTrue(np.allclose(m1, m2), msg.format(m1, m2))

        h = self.c1
        m1 = h.dot(d).toarray()
        m2 = array_dot_left(h, d)
        self.assertTrue(np.allclose(m1, m2), msg.format(m1, m2))

        h = self.a12
        m1 = h.dot(d).toarray()
        m2 = array_dot_left(h, d)
        self.assertTrue(np.allclose(m1, m2), msg.format(m1, m2))

        h = self.a12 + self.b21
        m1 = h.dot(d).toarray()
        m2 = array_dot_left(h, d)
        self.assertTrue(np.allclose(m1, m2), msg.format(m1, m2))


    def test_dot_right(self):

        def array_dot_right(left, right):
            assert isinstance(left, BaseTensor)
            assert isinstance(right, BaseTensor)
            assert left.y_dim == 0
            assert left.x_dim == right.n
            dim = max(right.dim, left.dim)
            p = right.x_dim + right.y_dim
            r_idx = [slice(None)] * (p + right.n) + [None] * left.n + \
                    [slice(None)] * right.dim + [None] * (dim - right.dim)
            l_idx = [None] * p + [slice(None)] * (left.x_dim + left.n) + \
                    [None] * (dim - left.dim)
            val = np.multiply(
                left.toarray()[tuple(l_idx)],
                right.toarray()[tuple(r_idx)])
            val = val.sum(tuple(p + np.arange(right.n)))
            return val

        n = 3
        msg = "\nM1=\n{}\n\nM2=\n{}\n"
        d = Tensor(np.arange(1, n ** 2 + 1).reshape((n, n))[None, None, :, :], 2, 0, 0, array('b', [0, 1]))

        h = self.a12
        m1 = d.dot(h).toarray()
        m2 = array_dot_right(d, h)
        self.assertTrue(np.allclose(m1, m2), msg.format(m1, m2))

        h = self.a12 + self.b21
        m1 = d.dot(h).toarray()
        m2 = array_dot_right(d, h)
        self.assertTrue(np.allclose(m1, m2), msg.format(m1, m2))


if __name__ == "__main__":
    unittest.main()
