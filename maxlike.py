import numpy as np
import pandas as pd


class IndexMap(list):
    def __init__(self, indexes):
        if isinstance(indexes, int):
            indexes = [indexes]
        assert min(indexes) >= 0
        self.extend(indexes)

    def __call__(self, params):
        return map(params.__getitem__, self)


class MaxLike(object):
    def __init__(self):
        # atomic param related
        self.params = []
        self.free = []
        self.label = []
        self.params_fixed = np.array([])

        # whole param related
        self.split_ = [0]
        self.fixed_map = np.array([], np.int)

        # model related
        self.model = None

        self.constraint = []
        self.reg = []

        self.features_names = []
        self.N = None

        self.g_last = None
        self.verbose = False

    def fit(self, observations):
        raise NotImplementedError

    def like(self, params):
        raise NotImplementedError

    def grad_like(self, params):
        raise NotImplementedError

    def hess_like(self, params):
        raise NotImplementedError

    def g(self, params):
        """
        Objective function, to maximize.
        """
        g = self.like(params)
        for param_map, gamma, h in self.reg:
            g += gamma * h(param_map(params))

        return g

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

        n_el = sum([el.size for el in self.free])
        self.params.append(values)
        self.label.append(label)
        self.split_.append(self.split_[-1] + values.size)

        # Fixed values
        self.params_fixed = np.concatenate((
            self.params_fixed, values[fixed]))

        # Map to insert fixed values
        self.free.append(~fixed)
        self.fixed_map = np.concatenate((
            self.fixed_map,
            (lambda x: x - np.arange(x.size))(
                (lambda x: np.arange(x.size)[x])(
                    fixed.flatten())) + n_el))

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
        k = sum([c.size for c in self.params]) - len(self.constraint)

        return 2 * k * (1 + (k - 1) / (self.N.sum() - k - 1)) - \
            2 * self.g(self.params)

    def bayesian_information_criterion(self):
        """
        Bayesian information criterion.
        (The best model is that which minimizes it)
        """
        # k: # of free parameters
        k = sum([c.size for c in self.params]) - len(self.constraint)

        return k * np.log(self.N.sum()) - 2 * self.g(self.params)

    def __reshape_array(self, flat_array, val=0):
        """
        Reshapes as array in order to have the same format as self.params

        Parameters
        ----------
        flat_array: ndarray
            Array with one axis with the same lenght as the number of free
            parameters.
        val: float, ndarray (optional)
            Value(s) to fill in fixed_map places. If it is a ndarray it
            needs to have the same size as fixed_map.
        """
        return [np.insert(flat_array, self.fixed_map, val)[
                self.split_[i]:self.split_[i + 1]].
                reshape(self.params[i].shape)
                for i in range(len(self.params))]

    def __reshape_params(self, params_free):
        return self.__reshape_array(params_free, self.params_fixed)

    def __reshape_matrix(self, matrix, value=np.NaN):
        if matrix.ndim != 2:
            raise ValueError("matrix.ndim != 2")
        elif matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix.shape[0] != matrix.shape[1]")

        n = len(self.params)

        # Insert columns and rows ith fixed values
        matrix = np.insert(matrix, self.fixed_map, value, axis=0)
        matrix = np.insert(matrix, self.fixed_map, value, axis=1)

        # Split by blocks
        matrix = [[matrix[self.split_[i]:self.split_[i + 1],
                   self.split_[j]:self.split_[j + 1]]
                   for j in range(n)] for i in range(n)]

        # Reshape each block accordingly
        matrix = [[matrix[i][j].reshape(np.concatenate((
            self.params[i].shape, self.params[j].shape)))
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
        hess_c = [[np.zeros(self.params[j].shape)
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

        hess = [[hess[i][j][np.multiply.outer(
            self.free[j], self.free[i])].reshape(
            (self.free[j].sum(), self.free[i].sum()))
            for j in range(i + 1)] for i in range(n)]
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
            if new_g >= self.g_last:
                self.params = new_params
                self.g_last = new_g
                self.__flat_hess = hess
                return True
            else:
                u *= .5
        if self.verbose:
            print("""Error: the objective function did not increased after %d
                     steps""" % max_steps)

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
            new_g = self.g_last
            self.__step()
            if self.verbose:
                print(i, new_g)
            if abs(new_g / self.g_last) - 1 < tol:
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
        # feature, labels, weights
        """
        Accepts a Series with a multiIndex - where the values are the labels
        and the multiIndex the features.

        Parameters
        ----------
        observations : Series
            Observation list.
        """
        axis = [level.sort_values().values for level in
                observations.index.levels]
        self.features_names = list(observations.index.names)
        shape = map(len, axis)
        df = observations.to_frame('X')
        df['N'] = 1  # frequency
        # sum repeated occurrences for the same index entry
        df = df.groupby(df.index).sum()
        df = df.reindex(pd.MultiIndex.from_product(axis)).fillna(0)
        self.X = df['X'].values.reshape(shape)
        self.N = df['N'].values.reshape(shape)
        return axis

    def like(self, params):
        """
        Likelihood function.
        """
        ln_y = self.model(params)
        return (self.X * ln_y - self.N * np.exp(ln_y)).sum()

    def grad_like(self, params):
        """
        Calculates the gradient of the log-likelihood function.
        """
        delta = self.X - self.N * np.exp(self.model(params))
        der = self.model.grad(params)
        return [(delta[[Ellipsis] + [None] * self.params[i].ndim] *
                 der[i]).sum(tuple(range(self.N.ndim)))
                for i in range(len(self.params))]

    def hess_like(self, params):
        """
        Calculates the hessian of the log-likelihood function.
        """
        hess = []
        ln_y = self.model(params)
        delta = self.X - self.N * np.exp(ln_y)
        grad = self.model.grad(params)
        slc_asis = []
        slc_bdct = []
        slc_feat = [slice(None)] * self.N.ndim

        for i in range(len(self.params)):
            slc_asis.append([slice(None)] * self.params[i].ndim)
            slc_bdct.append([None] * self.params[i].ndim)
            hess_line = []
            for j in range(i + 1):
                cube = self.model.hess(params, i, j)
                h = -((self.N * np.exp(ln_y)
                       )[slc_feat + slc_bdct[j] + slc_bdct[i]] *
                      grad[j][slc_feat + slc_asis[j] + slc_bdct[i]] *
                      grad[i][slc_feat + slc_bdct[j] + slc_asis[i]]
                      ).sum(tuple(np.arange(self.N.ndim)))
                h += (delta[slc_feat + slc_bdct[j] + slc_bdct[i]] * cube).sum(
                    tuple(np.arange(self.N.ndim)))
                hess_line.append(h)
            hess.append(hess_line)
        return hess
