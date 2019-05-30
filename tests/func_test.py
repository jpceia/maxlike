import numpy as np
import unittest
from maxlike.func import (
    Exp, Logistic, MarkovMatrix, Scalar, Sum, Product, X)


class Test(unittest.TestCase):

    def check_comm(self, foo, n=1, params_list=None):
        if params_list is None:
            if n == 1:
                params_list = [
                    [1],
                    [np.arange(3)]
                ]
            elif n == 2:
                params_list = [
                    [np.arange(3), np.ones(3)]
                ]
            else:
                raise ValueError
        for params in params_list:
            val, grad, hess = foo.eval(params)
            self.assertTrue(np.allclose(
                val.toarray(), foo(params).toarray()))
            for i, g in enumerate(grad):
                self.assertTrue(np.allclose(
                    g.toarray(), foo.grad(params, i).toarray()))
                for j, h in enumerate(hess[i]):
                    if j > i:
                        break
                    self.assertTrue(np.allclose(
                        h.toarray(),
                        foo.hess(params, i, j).toarray()))

    def test_exp(self):
        self.check_comm(Exp())

    def test_logistic(self):
        self.check_comm(Logistic())

    def test_affine(self):
        self.check_comm(2 * X() + 1)

    def test_sum(self):
        foo = Sum(2)
        foo.add(X(), 0, 0)
        foo.add(X(), 1, 1)
        self.check_comm(foo, 2)

    def test_product(self):
        foo = Product(2)
        foo.add(X(), 0, 0)
        foo.add(X(), 1, 1)
        self.check_comm(foo, 2)

    def test_markov_matrix(self):
        self.check_comm(
            MarkovMatrix(size=5),
            # params need to be strictly positive
            params_list=[
                [np.ones(3),
                 np.arange(3) + 1]]
            )


if __name__ == "__main__":
    unittest.main()
