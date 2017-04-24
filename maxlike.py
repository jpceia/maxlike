import numpy as np
import pandas as pd
from common import IndexMap, transpose
from scipy.misc import factorial


class MaxLike(object):
    def __init__(self):
        self.delete_param()

        # model related
        self.model = None

        self.constraint = []
        self.reg = []

        self.features_names = []
        self.N = None

        self.g_last = None
        self.verbose = False

    def fit(self, observations):
        # feature, labels, weights
        """
        Accepts a Series with a multiIndex - where the values are the labels
        and the multiIndex the features.

        Parameters
        ----------
        observations : Series
            Observation list.
        """
        raise NotImplementedError

    def like(self, params):
        """
        Likelihood function.
        """
        raise NotImplementedError

    def grad_like(self, params):
        """
        Calculates the gradient of the log-likelihood function.
        """
        raise NotImplementedError

    def hess_like(self, params):
        """
        Calculates the hessian of the log-likelihood function.
        """
        raise NotImplementedError

    def g(self, params):
        """
        Objective function, to maximize.
        """
        g = self.like(params)
        for param_map, gamma, h in self.reg:
            g += gamma * h(param_map(params))

        return g / self.N.sum()

    def delete_param(self, i=None):
        if i is None:
            self.params = []
            self.free = []
            self.label = []
        else:
            del self.params[i]
            del self.free[i]
            del self.label[i]
            # need to correct model, reg and constraint also 

    def add_param(self, values, fixed=None, label=''):
        """
        Adds a new parameter.

        Parameters
        ----------
        values: list
            Initial guess for the parameter.
        fixed: list (optional)
            Boolean arrays with value = True if the parameter has a
            fixed value.
        label: string (optional)
            label for that parameter.
        """
        if isinstance(values, (int, float, tuple, list)):
            values = np.array(values)
        elif not isinstance(values, np.ndarray):
            raise TypeError

        if fixed is None:
            fixed = np.zeros(values.shape, np.bool)
        elif isinstance(fixed, bool):
            fixed *= np.ones(values.shape, np.bool)
        elif isinstance(fixed, (tuple, list)):
            fixed = np.array(fixed)
        elif not isinstance(fixed, np.ndarray):
            raise TypeError

        if not fixed.shape == values.shape:
            raise ValueError("shape mismatch")

        self.params.append(values)
        self.label.append(label)

        # Fixed values
        self.params_fixed = np.concatenate((
            self.params_fixed, values[fixed]))

        # Map to insert fixed values
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
            2 * self.g(self.params)

    def bayesian_information_criterion(self):
        """
        Bayesian information criterion.
        (The best model is that which minimizes it)
        """
        # k: # of free parameters
        k = sum([f.sum() for f in self.free]) - len(self.constraint)
        return k * np.log(self.N.sum()) - 2 * self.g(self.params)

    def _sum_feat(self):
        return tuple(-np.arange(self.N.ndim) - 1)

    def __reshape_array(self, flat_array, val=0):
        """
        Reshapes as array in order to have the same format as self.params

        Parameters
        ----------
        flat_array: ndarray
            Array with one axis with the same lenght as the number of free
            parameters.
        val: float, ndarray (optional)
        """
        shaped_array = []
        s_0 = 0
        f_0 = 0
        for i in range(len(self.params)):
            s_1 = s_0 + self.free[i].sum()
            f_1 = f_0 + (~self.free[i]).sum()
            val_map = (lambda x: x - np.arange(x.size))(
                      (lambda x: np.arange(x.size)[x])(
                          (~self.free[i]).flatten()))
            shaped_array.append(
                np.insert(
                    flat_array[s_0:s_1], val_map, val[f_0:f_1]).
                reshape(self.free[i].shape))
            s_0 = s_1
            f_0 = f_1

        return shaped_array

    def __reshape_params(self, params_free):
        return self.__reshape_array(
            params_free,
            np.concatenate([self.params[~self.free[i]]
                            for i in range(len(self.free))]))

    def __reshape_matrix(self, matrix, val=np.NaN):
        if matrix.ndim != 2:
            raise ValueError("matrix.ndim != 2")
        elif matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix.shape[0] != matrix.shape[1]")

        n = len(self.params)

        # Split by blocks
        s_ = [0]
        f_ = [0]
        val_map = []
        for i in range(n):
            s_.append(s_[-1] + self.free[i].size)
            f_.append(f_[-1] + (~self.free[i]).sum())
            val_map.append((lambda x: x - np.arange(x.size))(
                (lambda x: np.arange(x.size)[x])(
                    (~self.free[i]).flatten())))

        matrix = [[np.insert(np.insert(
                   matrix[s_[i]:s_[i + 1], s_[j]:s_[j + 1]],
                   val_map[i], val[f_[i]:f_[i + 1]]),
                   val_map[j], val[f_[j]:f_[j + 1]]).reshape(
                       np.concatenate(self.free[i].shape, self.free[j].shape))
                   for j in range(n)] for i in range(n)]
        return matrix

    def fisher_matrix(self):
        return self.__reshape_matrix(self.__flat_hess, 0)

    def error_matrix(self):
        """
        Covariance Matrix.
        """
        return self.__reshape_matrix(-np.linalg.inv(self.__flat_hess))

    def std_error(self):
        """
        Standard Deviation Array.
        """
        cut = -len(self.constraint)
        if cut == 0:
            cut = None
        return self.__reshape_array(np.sqrt(np.diag(-np.linalg.inv(
            self.__flat_hess))[:cut]))

    def __step(self, max_steps=20):

        n = len(self.params)
        c_len = len(self.constraint)

        # --------------------------------------------------------------------
        # 1st phase: Evaluate and sum
        # --------------------------------------------------------------------
        grad = self.grad_like(self.params) + [0] * c_len
        hess = self.hess_like(self.params)

        # Add blocks corresponding to constraint variables:
        # Hess_lambda_params = grad_g
        hess_c = [[np.zeros(self.free[j].shape)
                   for j in range(n)] for i in range(c_len)]

        # --------------------------------------------------------------------
        # 2nd phase: Add constraints / regularization to grad and hess
        # --------------------------------------------------------------------
        count = -1
        for param_map, gamma, g in self.constraint:
            count += 1
            args = param_map(self.params)
            grad_g = g.grad(args)
            hess_g = g.hess(args)
            for i in range(len(param_map)):
                idx = param_map[i]
                grad[idx] += gamma * grad_g[i]
                grad[n + count] = np.array([g(args)])
                hess_c[count][idx] += grad_g[i]
                for j in range(i + 1):
                    hess[idx][param_map[j]] += gamma * hess_g[i][j]

        for param_map, gamma, h in self.reg:
            args = param_map(self.params)
            grad_h = h.grad(args)
            hess_h = h.hess(args)
            for i in range(len(param_map)):
                idx = param_map[i]
                grad[idx] -= gamma * grad_h[i]
                for j in range(i + 1):
                    hess[idx][param_map[j]] -= gamma * hess_h[i][j]

        # --------------------------------------------------------------------
        # 3rd phase: Reshape and flatten
        # --------------------------------------------------------------------
        params = [self.params[i][self.free[i]] for i in range(n)]
        grad = [grad[i][self.free[i]] for i in range(n)] + grad[n:]

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
        hess_c = [[hess_c[i][j][self.free[j]] for j in range(n)]
                  for i in range(c_len)]

        # --------------------------------------------------------------------
        # 4th phase: Consolidate blocks
        # --------------------------------------------------------------------
        params = np.concatenate(params)
        grad = np.concatenate(grad)
        hess = np.vstack(map(np.hstack, hess))

        if c_len > 0:
            hess_c = np.vstack(map(np.concatenate, hess_c))
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
            new_params = self.__reshape_params(params + u * d)
            new_g = self.g(new_params)
            if new_g - self.g_last >= 0:
                self.params = new_params
                self.g_last = new_g
                self.__flat_hess = hess
                return True
            else:
                u *= .5
        raise RuntimeError("Error: the objective function did not increased",
                           "after %d steps" % max_steps)

    def run(self, tol=1e-8, max_steps=100):
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
        self.g_last = self.g(self.params)
        for i in range(max_steps):
            old_g = self.g_last
            self.__step()
            if self.verbose:
                print(i, self.g_last)
            if abs(old_g / self.g_last - 1) < tol:
                return True
        return False


class Poisson(MaxLike):
    """
    Class to model data under a Poisson Regression.
    """

    def __init__(self):
        super(Poisson, self).__init__()
        self.X = None

    def fit(self, observations):
        axis = [level.sort_values().values for level in
                observations.index.levels]
        self.features_names = list(observations.index.names)
        shape = map(len, axis)
        df = observations.groupby(observations.index).agg({
            'N': np.size,
            'X': np.sum})
        df = df.reindex(pd.MultiIndex.from_product(axis)).fillna(0)
        self.X = df['X'].values.reshape(shape)
        self.N = df['N'].values.reshape(shape)
        return axis

    def like(self, params):
        ln_y = self.model(params)
        return (self.X * ln_y - self.N * np.exp(ln_y) -
                np.log(factorial(self.X))).sum()

    def grad_like(self, params):
        delta = self.X - self.N * np.exp(self.model(params))
        der = self.model.grad(params)
        return [(delta * der[i]).sum(self._sum_feat())
                for i in range(len(self.params))]

    def hess_like(self, params):
        Y = self.N * np.exp(self.model(params))
        delta = self.X - Y
        grad = self.model.grad(params)
        return [[(delta * self.model.hess(params, i, j) -
                  Y * grad[i] * transpose(grad, params, j, i)).sum(
                 self._sum_feat())
                 for j in range(i + 1)] for i in range(len(self.params))]
