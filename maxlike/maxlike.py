import numpy as np
import abc
from .common import IndexMap, transpose, vector_sum, matrix_sum
from scipy.misc import factorial


class MaxLike(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        self.reset_params()

        # model related
        self.model = None

        self.constraint = []
        self.reg = []

        self.N = None

        self.g_last = None

    @abc.abstractmethod
    def like(self, params):
        """
        Likelihood function.
        """
        pass

    @abc.abstractmethod
    def grad_like(self, params):
        """
        Calculates the gradient of the log-likelihood function.
        """
        pass

    @abc.abstractmethod
    def hess_like(self, params):
        """
        Calculates the hessian of the log-likelihood function.
        """
        pass

    def g(self, params):
        """
        Objective function, to maximize.
        """
        g = self.like(params)
        for param_map, gamma, h in self.reg:
            g -= gamma * h(param_map(params))

        return g / self.N.sum()

    def reset_params(self):
        self.param_ = []
        self.free = []

    def add_param(self, values, fixed=None):
        """
        Adds a new parameter.

        Parameters
        ----------
        values: list
            Initial guess for the parameter.
        fixed: list (optional)
            Boolean arrays with value = True if the parameter has a
            fixed value.
        """
        if isinstance(values, (int, float, tuple, list)):
            values = np.asarray(values)
        elif not isinstance(values, np.ndarray):
            raise TypeError

        if fixed is None:
            fixed = np.zeros(values.shape, np.bool)
        elif isinstance(fixed, bool):
            fixed *= np.ones(values.shape, np.bool)
        elif isinstance(fixed, (tuple, list)):
            fixed = np.asarray(fixed)
        elif not isinstance(fixed, np.ndarray):
            raise TypeError

        if not fixed.shape == values.shape:
            raise ValueError("shape mismatch")

        self.param_.append(values)
        self.free.append(~fixed)

    def add_constraint(self, param_map, g):
        """
        Adds a constraint factor to the objective function.

        Parameters
        ----------
        param_map : int, list
            index of the parameters to witch g applies
        g : func
            Constraint function.
        """
        self.constraint.append((IndexMap(param_map), 0, g))

    def add_regularization(self, param_map, gamma, h):
        """
        Adds a regularization factor to the objective function.

        Parameters
        ----------
        param_map : int, list
            index of the parameters to witch f applies
        gamma : float
            Scale parameter to apply to f
        h : func
            Regularization function
        """
        self.reg.append((IndexMap(param_map), gamma, h))

    def akaine_information_criterion(self):
        """
        Akaike information criterion.
        (The best model is that which minimizes it)
        """
        # k: # of free parameters
        k = sum([f.sum() for f in self.free]) - len(self.constraint)
        return 2 * k * (1 + (k - 1) / (self.N.sum() - k - 1)) - \
            2 * self.g(self.param_)

    def bayesian_information_criterion(self):
        """
        Bayesian information criterion.
        (The best model is that which minimizes it)
        """
        # k: # of free parameters
        k = sum([f.sum() for f in self.free]) - len(self.constraint)
        return k * np.log(self.N.sum()) - 2 * self.g(self.param_)

    def _sum_feat(self):
        return tuple(-np.arange(self.N.ndim) - 1)

    def __reshape_array(self, flat_array, val=0):
        """
        Reshapes as array in order to have the same format as self.param_

        Parameters
        ----------
        flat_array: ndarray
            Array with one axis with the same lenght as the number of free
            parameters.
        val: float, ndarray (optional)
        """
        def fix_map(arr):
            return (lambda x: x - np.arange(x.size))(
                (lambda x: np.arange(x.size)[x])(arr))

        shaped_array = []
        s_0 = 0

        # val is a scalar
        if isinstance(val, (int, float)):
            for i, free in enumerate(self.free):
                s_1 = s_0 + free.sum()
                shaped_array.append(np.insert(
                    flat_array[s_0:s_1],
                    fix_map((~free).flatten()),
                    val).reshape(free.shape))
                s_0 = s_1
        # val is an array
        else:
            f_0 = 0
            for i, free in enumerate(self.free):
                s_1 = s_0 + free.sum()
                f_1 = f_0 + (~free).sum()
                shaped_array.append(np.insert(
                    flat_array[s_0:s_1],
                    fix_map((~free).flatten()),
                    val[f_0:f_1]).reshape(free.shape))
                s_0 = s_1
                f_0 = f_1
        return shaped_array

    def __reshape_params(self, params_free):
        return self.__reshape_array(
            params_free,
            np.concatenate([param[~free]
                            for param, free in zip(self.param_, self.free)]))

    def __reshape_matrix(self, matrix, val=np.NaN):
        if matrix.ndim != 2:
            raise ValueError("matrix.ndim != 2")
        elif matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix.shape[0] != matrix.shape[1]")

        n = len(self.param_)

        # Split by blocks
        s_ = [0]
        f_ = [0]
        val_map = []
        for i, free in enumerate(self.free):
            s_.append(s_[-1] + free.size)
            f_.append(f_[-1] + (~free).sum())
            val_map.append((lambda x: x - np.arange(x.size))(
                (lambda x: np.arange(x.size)[x])(
                    (~free).flatten())))

        # val is a scalar
        if isinstance(val, (int, float)):
            return [[np.insert(np.insert(
                matrix[s_[i]:s_[i + 1], s_[j]:s_[j + 1]],
                val_map[i], val), val_map[j], val).reshape(
                self.param_[i].shape + self.param_[j].shape)
                for j in range(n)] for i in range(n)]
        else:
            return [[np.insert(np.insert(
                matrix[s_[i]:s_[i + 1], s_[j]:s_[j + 1]],
                val_map[i], val[f_[i]:f_[i + 1]]),
                val_map[j], val[f_[j]:f_[j + 1]]).reshape(
                self.param_[i].shape + self.param_[j].shape)
                for j in range(n)] for i in range(n)]

    def fisher_matrix(self):
        return self.__reshape_matrix(self.flat_hess_, 0)

    def error_matrix(self):
        """
        Covariance Matrix.
        """
        return self.__reshape_matrix(-np.linalg.inv(self.flat_hess_))

    def std_error(self):
        """
        Standard Deviation Array.
        """
        cut = -len(self.constraint)
        if cut == 0:
            cut = None
        return self.__reshape_array(np.sqrt(np.diag(-np.linalg.inv(
            self.flat_hess_))[:cut]))

    def __step(self):
        max_steps = 20
        n = len(self.param_)
        c_len = len(self.constraint)

        # --------------------------------------------------------------------
        # 1st phase: Evaluate and sum
        # --------------------------------------------------------------------
        grad = self.grad_like(self.param_) + [0] * c_len
        hess = self.hess_like(self.param_)

        # Add blocks corresponding to constraint variables:
        # Hess_lambda_params = grad_g
        hess_c = [[np.zeros(free.shape)
                   for free in self.free]
                  for _ in range(c_len)]

        # --------------------------------------------------------------------
        # 2nd phase: Add constraints / regularization to grad and hess
        # --------------------------------------------------------------------
        count = -1
        for param_map, gamma, g in self.constraint:
            count += 1
            args = param_map(self.param_)
            grad_g = g.grad(args)
            hess_g = g.hess(args)
            for i, idx in enumerate(param_map):
                grad[idx] += gamma * grad_g[i]
                grad[n + count] = np.asarray([g(args)])
                hess_c[count][idx] += grad_g[i]
                for j in range(i + 1):
                    hess[idx][param_map[j]] += gamma * hess_g[i][j]

        for param_map, gamma, h in self.reg:
            args = param_map(self.param_)
            grad_h = h.grad(args)
            hess_h = h.hess(args)
            for i, idx in enumerate(param_map):
                grad[idx] -= gamma * grad_h[i]
                for j in range(i + 1):
                    hess[idx][param_map[j]] -= gamma * hess_h[i][j]

        # --------------------------------------------------------------------
        # 3rd phase: Reshape and flatten
        # --------------------------------------------------------------------
        flat_params = [param[free] for param, free in zip(self.param_, self.free)]
        grad = [grad[i][free] for i, free in enumerate(self.free)] + grad[n:]

        # ------
        # | aa |
        # -----------
        # | ab | bb |
        # ----------------
        # | ac | bc | cc |
        # ----------------
        #
        # then hess[i][j].shape = shape[j] x shape[i]

        hess = [[hess[j][i][np.multiply.outer(
            self.free[i], self.free[j])].reshape(
            (self.free[i].sum(), self.free[j].sum()))
            for i in range(j + 1)] for j in range(n)]
        hess = [[hess[i][j].transpose() for j in range(i)] +
                [hess[j][i] for j in range(i, n)] for i in range(n)]
        hess_c = [[hess_c[i][j][free] for j, free in enumerate(self.free)]
                  for i in range(c_len)]

        # --------------------------------------------------------------------
        # 4th phase: Consolidate blocks
        # --------------------------------------------------------------------
        flat_params = np.concatenate(flat_params)
        grad = np.concatenate(grad)
        hess = np.vstack(list(map(np.hstack, hess)))

        if c_len > 0:
            hess_c = np.vstack(list(map(np.concatenate, hess_c)))
            hess = np.vstack([
                np.hstack([hess, hess_c.transpose()]),
                np.hstack([hess_c,
                           np.zeros((c_len, c_len))])])

        # --------------------------------------------------------------------
        # 5th phase: Compute
        # --------------------------------------------------------------------
        u = 1
        d = -np.linalg.solve(hess, grad)
        if c_len > 0:
            d = d[:-c_len]

        # change = np.linalg.norm(d) / np.linalg.norm(params)
        for i in range(max_steps):
            new_params = self.__reshape_params(flat_params + u * d)
            new_g = self.g(new_params)
            if new_g - self.g_last >= 0:
                self.param_ = new_params
                self.g_last = new_g
                self.flat_hess_ = hess
                return None
            else:
                u *= .5
        raise RuntimeError("Error: the objective function did not increase",
                           "after %d steps" % max_steps)

    def fit(self, tol=1e-8, max_steps=100, verbose=False, **kwargs):
        """
        Run the algorithm to find the best fit.

        Parameters
        ----------
        tol : float (optional)
            Tolerance for termination.
        max_steps : int (optional)
            Maximum number of steps that the algorithm will perform.

        Returns
        -------
        out : boolean
            Returns True if the algorithm converges, otherwise returns
            False.
        """
        for k, v in kwargs.items():
            self.__dict__[k] = v

        self.g_last = self.g(self.param_)
        for i in range(max_steps):
            old_g = self.g_last
            self.__step()
            if verbose:
                print(i, self.g_last)
            if abs(old_g / self.g_last - 1) < tol:
                return None
        raise RuntimeError("Error: the objective function did not converge",
                           "after %d steps" % max_steps)


class Poisson(MaxLike):
    """
    Class to model data under a Poisson Regression.
    """

    def __init__(self):
        MaxLike.__init__(self)
        self.X = None

    def like(self, params):
        ln_y = self.model(params)
        return (self.X * ln_y - self.N * np.exp(ln_y) -
                np.log(factorial(self.X))).sum()

    def grad_like(self, params):
        delta = self.X - self.N * np.exp(self.model(params))
        return [vector_sum(
                delta * d,
                params, self.model.param_feat, i)
                for i, d in enumerate(self.model.grad(params))]

    def hess_like(self, params):
        y = self.N * np.exp(self.model(params))
        delta = self.X - y
        der = self.model.grad(params)
        return [[matrix_sum(
                 delta * self.model.hess(params, i, j) -
                 y * d * transpose(der, params, j, i),
                 params, self.model.param_feat, i, j)
                 for j in range(i + 1)] for i, d in enumerate(der)]


class Logistic(MaxLike):
    """
    Class to model under a Logistic Regression
    """

    def __init__(self):
        MaxLike.__init__(self)

    def like(self, params):
        y = self.model(params)
        return -(self.N * np.log(1 + np.exp(-y)) + (self.N - self.P) * y).sum()

    def grad_like(self, params):
        delta = self.P - (self.N / (1 + np.exp(-self.model(params))))
        return [vector_sum(delta * d,
                           params, self.model.param_feat, i)
                for i, d in enumerate(self.model.grad(params))]

    def hess_like(self, params):
        p = 1 / (1 + np.exp(-self.model(params)))
        der = self.model.grad(params)
        delta = self.P - p * self.N
        delta2 = - p * (1 - p) * self.N
        return[[matrix_sum(
                delta * self.model.hess(params, i, j) +
                delta2 * der[i] * transpose(der, params, j, i),
                params, self.model.param_feat, i, j)
                for j in range(i + 1)] for i, d in enumerate(der)]


class Normal(MaxLike):
    """
    Class to model data under a Normal Distribution Regression.
    """

    def __init__(self):
        MaxLike.__init__(self)
        self.U = None
        self.V = None

    def like(self, params):
        # L: -v -.5*exp(-2v)*(x-u)^2
        u = self.model(params, k=0)
        v = self.model(params, k=1)
        return -(v / np.square(self.N) + .5 * np.exp(-2 * v) *
                 (np.square(self.V) + np.square(self.U - u))).sum()

    def grad_like(self, params):
        # dLdv :  (-1 + exp(-2v)*(x-u)^2)
        # dLdu : exp(-2v)*(x-u)
        u = self.model(params, k=0)
        v = self.model(params, k=1)
        grad_u = self.model.grad(params, k=0)
        grad_v = self.model.grad(params, k=1)
        dLdv = -1 / np.square(self.N) + np.exp(-2 * v) * \
            (np.square(self.V) + np.square(self.U - u))
        return [vector_sum(dLdv * grad_v[i] +
                           np.exp(-2 * v) * (self.U - u) * grad_u[i],
                           params, self.model.param_feat, i)
                for i in range(len(params))]

    def hess_like(self, params):
        u = self.model(params, k=0)
        v = self.model(params, k=1)
        grad_u = self.model.grad(params, k=0)
        grad_v = self.model.grad(params, k=1)
        dLdv = -1 / np.square(self.N) + np.exp(-2 * v) * \
            (np.square(self.V) + np.square(self.U - u))
        return [[matrix_sum(dLdv * self.model.hess(params, i, j, k=1) -
                 2 * np.exp(-2 * v) * (
                     (np.square(self.V) + np.square(self.U - u)) *
                     grad_v[i] * transpose(grad_v, params, j, i) +
                     (self.U - u) * (
                         grad_v[i] * transpose(grad_u, params, j, i) +
                         grad_u[i] * transpose(grad_v, params, j, i) -
                         .5 * self.model.hess(params, i, j, k=0)
                     ) + .5 * grad_u[i] * transpose(grad_u, params, j, i)),
                 params, self.model.param_feat, i, j)
                 for j in range(i + 1)] for i in range(len(params))]


class Finite(MaxLike):
    """
    Class to model an probabilistic regression under an arbitrary
    Discrete Finite Distribution
    """

    def __init__(self):
        MaxLike.__init__(self)

    def like(self, params):
        # N * ln p
        return (self.N * np.log(self.model(params))).sum()

    def grad_like(self, params):
        p = self.model(params)
        der = self.model.grad(params)
        delta = self.N / p
        return [vector_sum(delta * der[i], params, self.model.param_feat, i)
                for i in range(len(params))]

    def hess_like(self, params):
        p = self.model(params)
        der = self.model.grad(params)
        delta = self.N / p
        return [[matrix_sum(delta *
                            (self.model.hess(params, i, j) -
                             der[i] * transpose(der, params, j, i) / p),
                            params, self.model.params_feat, i, j)
                 for j in range(i + 1)] for i in range(len(params))]
