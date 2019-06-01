import numpy as np
import unittest
from maxlike.func import (
    Exp, Logistic, ArcTan, Scalar, Sum, Product, X,
    MarkovMatrix, Poisson, LogNegativeBinomial)


class Test(unittest.TestCase):

    def check_comm(self, foo, params):
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


    def check_comm_run(self, foo, n=1, params_list=None):
        if n == 1:
            params_list = [[1], [np.arange(3)]]
        elif n == 2:
            params_list = [[np.arange(3), np.ones(3)]]
        else:
            raise NotImplementedError

        for params in params_list:
            self.check_comm(foo, params)

    def test_exp(self):
        self.check_comm_run(Exp())

    def test_add(self):
        self.check_comm_run(X() + 1)

    def test_radd(self):
        self.check_comm_run(1 + X())

    def test_sub(self):
        self.check_comm_run(X() - 1)

    def test_rsub(self):
        self.check_comm_run(1 - X())

    def test_mul(self):
        self.check_comm_run(X() * 2)

    def test_rmul(self):
        self.check_comm_run(2 * X())

    def test_sum(self):
        foo = Sum(2)
        foo.add(X(), 0, 0)
        foo.add(X(), 1, 1)
        self.check_comm_run(foo, 2)

    def test_product(self):
        foo = Product(2)
        foo.add(X(), 0, 0)
        foo.add(X(), 1, 1)
        self.check_comm_run(foo, 2)

    def test_poisson(self):
        self.check_comm_run(Poisson())

    def test_markov_matrix(self):
        self.check_comm(
            MarkovMatrix(size=5),
            # params need to be strictly positive
            [np.ones(3), np.arange(3) + 1])

    def test_log_negative_binomial(self):
        self.check_comm_run(LogNegativeBinomial() @ Exp())

    def test_logistic(self):
        self.check_comm_run(Logistic())

    def test_arctan(self):
        self.check_comm_run(ArcTan())


if __name__ == "__main__":
    unittest.main()
