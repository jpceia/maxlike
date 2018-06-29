import unittest
import sys
sys.path.insert(0, "..")
from maxlike.tensor import *


class Test(unittest.TestCase):

    def check_comm(self, foo, *t_list):
        a_list = [t.toarray() for t in t_list]
        M1 = foo(*a_list)
        M2 = foo(*t_list).toarray()
        np.testing.assert_array_equal(M1, M2)

    def test_simple_tensor(self):
        a = np.arange(1, 5).reshape((2, 2))
        b = np.arange(5, 9).reshape((2, 2))
        A, B = Tensor(a), Tensor(b)
        A1 = Tensor(a[None, :, :], 1, 0, 0, [0])
        B1 = Tensor(b[None, :, :], 1, 0, 0, [0])
        B2 = Tensor(b[None, :, :], 1, 0, 0, [1])
        A12 = Tensor(a[None, None, :, :], 1, 1, 0, [0], [1])
        B12 = Tensor(a[None, None, :, :], 1, 1, 0, [1], [0])

        self.check_comm(lambda x: -x, A)
        self.check_comm(lambda x: -x, A1)
        self.check_comm(lambda x: -x, A12)

        self.check_comm(lambda x, y: x + y, A, B)
        self.check_comm(lambda x, y: x - y, A, B)
        self.check_comm(lambda x, y: x * y, A, B)
        self.check_comm(lambda x, y: x / y, A, B)
        self.check_comm(lambda x, y: -(x + y), A, B)

        self.check_comm(lambda x, y: x + y, A, B1)
        self.check_comm(lambda x, y: x - y, A, B1)
        self.check_comm(lambda x, y: x * y, A, B1)
        self.check_comm(lambda x, y: -(x + y), A, B1)

        self.check_comm(lambda x, y: x + y, A1, B)
        self.check_comm(lambda x, y: x - y, A1, B)
        self.check_comm(lambda x, y: x * y, A1, B)
        self.check_comm(lambda x, y: x / y, A1, B)
        self.check_comm(lambda x, y: -(x + y), A1, B)

        self.check_comm(lambda x, y: x + y, A1, B1)
        self.check_comm(lambda x, y: x - y, A1, B1)
        self.check_comm(lambda x, y: x * y, A1, B1)
        self.check_comm(lambda x, y: -(x + y), A1, B1)

        self.check_comm(lambda x, y: x + y, A1, B2)
        self.check_comm(lambda x, y: x - y, A1, B2)
        self.check_comm(lambda x, y: x * y, A1, B2)
        self.check_comm(lambda x, y: -(x + y), A1, B2)

        self.check_comm(lambda x, y: x + y, A12, B12)
        self.check_comm(lambda x, y: x - y, A12, B12)
        self.check_comm(lambda x, y: x * y, A12, B12)
        self.check_comm(lambda x, y: -(x + y), A12, B12)

        #self.check_comm(lambda x, y: x + y, A1, B12)
        #self.check_comm(lambda x, y: x - y, A1, B12)
        #self.check_comm(lambda x, y: x * y, A1, B12)
        self.check_comm(lambda x, y: -(x + y), A1, B12)

        #self.check_comm(lambda x, y: x + y, A12, B1)
        #self.check_comm(lambda x, y: x - y, A12, B1)
        #self.check_comm(lambda x, y: x * y, A12, B1)
        #self.check_comm(lambda x, y: -(x + y), A12, B1)

        self.check_comm(lambda x, y: x + y, A12, B)
        self.check_comm(lambda x, y: x - y, A12, B)
        self.check_comm(lambda x, y: x * y, A12, B)
        self.check_comm(lambda x, y: x / y, A12, B)
        self.check_comm(lambda x, y: -(x + y), A12, B)

        self.check_comm(lambda x, y: x + y, A, B12)
        self.check_comm(lambda x, y: x - y, A, B12)
        self.check_comm(lambda x, y: x * y, A, B12)

    def test_generic_tensor(self):
        a = np.arange(1, 5).reshape((2, 2))
        b = np.arange(5, 9).reshape((2, 2))
        A, B = Tensor(a), Tensor(b)
        A1 = Tensor(a[None, :, :], 1, 0, 0, [0])
        A2 = Tensor(a[None, :, :], 1, 0, 0, [1])
        B1 = Tensor(b[None, :, :], 1, 0, 0, [0])
        B2 = Tensor(b[None, :, :], 1, 0, 0, [1])
        A12 = Tensor(a[None, None, :, :], 1, 1, 0, [0], [1])
        B12 = Tensor(a[None, None, :, :], 1, 1, 0, [1], [0])


        self.check_comm(lambda x, y, z: x + (y + z), A, B1, B2)
        self.check_comm(lambda x, y, z: x - (y + z), A, B1, B2)
        self.check_comm(lambda x, y, z: x * (y + z), A, B1, B2)

        self.check_comm(lambda x, y, z: x + (y + z), A1, B1, B2)
        self.check_comm(lambda x, y, z: x - (y + z), A1, B1, B2)
        self.check_comm(lambda x, y, z: x * (y + z), A1, B1, B2)

        #self.check_comm(lambda x, y, z: x + (y + z), A12, B1, B2)
        #self.check_comm(lambda x, y, z: x - (y + z), A12, B1, B2)
        #self.check_comm(lambda x, y, z: x * (y + z), A12, B1, B2)


        self.check_comm(lambda x, y, z: (x + y) + z, A1, A2, B)
        self.check_comm(lambda x, y, z: (x + y) - z, A1, A1, B)
        self.check_comm(lambda x, y, z: (x + y) * z, A1, A2, B)
        self.check_comm(lambda x, y, z: (x + y) / z, A1, A2, B)

        self.check_comm(lambda x, y, z: (x + y) + z, A1, A2, B1)
        self.check_comm(lambda x, y, z: (x + y) - z, A1, A1, B1)
        self.check_comm(lambda x, y, z: (x + y) * z, A1, A2, B1)

        # self.check_comm(lambda x, y, z: (x + y) + z, A1, A2, B12)
        # self.check_comm(lambda x, y, z: (x + y) - z, A1, A2, B12)
        # self.check_comm(lambda x, y, z: (x + y) * z, A1, A2, B12)
        

if __name__ == "__main__":
    unittest.main()