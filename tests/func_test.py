import numpy as np
import unittest
from maxlike.func import (
    Exp, Sigmoid, Scalar, Sum, Product, Affine, X, Quadratic, Vector,
    Collapse,
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


    def check_fd(self, foo, params, e=1e-6, atol=1e-6):
        """
        Central finite differences: value -> grad and grad -> hess
        (hess[i][j] has the axes of params[i] first, then params[j])
        """
        def dense(t):
            return np.asarray(t.toarray(), dtype=float)

        def num_jac(fun, i):
            jac = []
            for k in range(params[i].size):
                bump = np.zeros(params[i].size)
                bump[k] = e
                bump = bump.reshape(params[i].shape)
                up = [p + bump if l == i else p
                      for l, p in enumerate(params)]
                dn = [p - bump if l == i else p
                      for l, p in enumerate(params)]
                jac.append((dense(fun(up)) - dense(fun(dn))) / (2 * e))
            jac = np.stack(jac)
            return jac.reshape(params[i].shape + jac.shape[1:])

        def assert_close(a, b):
            a, b = np.broadcast_arrays(dense(a), b)
            self.assertTrue(np.allclose(a, b, atol=atol))

        val, grad, hess = foo.eval(params)
        for i in range(len(params)):
            assert_close(grad[i], num_jac(foo, i))
            for j in range(i + 1):
                assert_close(
                    hess[i][j], num_jac(lambda q: foo.grad(q, j), i))

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

    def test_sigmoid(self):
        self.check_comm_run(Sigmoid("Logistic"))
        self.check_comm_run(Sigmoid("ArcTan"))

    def test_affine(self):
        foo = Affine(X(), 1, 2)  # X.hess = null
        self.check_comm_run(foo)

        foo = Affine(Affine(X(), 2, 1), 1, 2)
        self.check_comm_run(foo)

    def test_compose_multi_input_outer(self):
        # Outer function with several vector inputs, each inner function
        # depending on a different parameter: the gradient of g_k with
        # respect to the other parameter is the null tensor.
        g1 = Sum(1)
        g1.add(Exp(), 0, 0)
        g2 = Sum(1)
        g2.add(Sigmoid("Logistic"), 1, 0)
        f = Product(1)
        f.add(X(), 0, 0)
        f.add(X(), 1, 0)
        foo = f @ [g1, g2]
        params = [np.array([.1, -.2, .3]), np.array([.2, .4, -.1])]
        self.check_comm(foo, params)

        self.check_fd(foo, params)

    def test_compose_dense_outer(self):
        # Outer function whose gradient is dense (no diagonal axes) while
        # the inner derivatives are diagonal: the diagonals have to be made
        # explicit by the chain rule.
        a = np.array([.1, -.2, .3])
        b = np.array([.2, .4, -.1, .5])
        m = np.arange(6.).reshape((3, 2)) / 10
        foo = Quadratic(.1, 2.) @ Exp()
        self.check_fd(foo, [a])
        self.check_fd(foo, [m])

        s = Sum(2)
        s.add(X(), 0, 0)
        s.add(-X(), 1, 1)
        self.check_fd(foo @ s, [a, b])

        # the same parameter on two feature axes
        s = Sum(2)
        s.add(X(), 0, 0)
        s.add(-X(), 0, 1)
        self.check_fd(foo @ s, [a])

    def test_collapse_partial(self):
        # summing only some of the feature axes keeps the other diagonals
        a = np.array([.1, -.2, .3])
        b = np.array([.2, .4, -.1, .5])
        h = np.array(.3)
        s = Sum(3)
        s.add(X(), 0, 0)
        s.add(-X(), 1, 1)
        s.add(Vector(np.arange(2) - .5), 2, 2)
        for sum_val in [0, [0], [1], [1, 2], [0, 2], [0, 1, 2]]:
            self.check_fd(Collapse(Exp() @ s, sum_val=sum_val), [a, b, h])

        # derivatives that are sums of incompatible diagonals
        s = Sum(2)
        s.add(X(), 0, 0)
        s.add(X(), 1, 1)
        s.add(Exp(), 0, 1)
        for sum_val in [[0], [1]]:
            self.check_fd(
                Collapse(Exp() @ s, sum_val=sum_val), [a, a + .1])

    def test_vector_shape_error(self):
        from maxlike.func import Vector
        with self.assertRaisesRegex(ValueError, "scalar or have the shape"):
            Vector(np.ones(3)).grad([np.ones((2, 1))], 0)


if __name__ == "__main__":
    unittest.main()
