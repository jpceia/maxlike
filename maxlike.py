import numpy as np
import pandas as pd


class func:
    def __init__(self):
        self.params = []

    def eval(self, coef, *args, **kargs):
        raise NotImplementedError

    def grad(self, coef, *args, **kargs):
        raise NotImplementedError

    def hess(self, coef, *args, **kargs):
        raise NotImplementedError

    @staticmethod
    def _wrap(param):
        if isinstance(param, (tuple, list)):
            return param
        else:
            return [param]


class linear(func):
    def __init__(self):
        self.weight = []
        self.level = 0

    def eval(self, coef):
        coef = self._wrap(coef)
        return sum([sum(self.weight[i] * coef[i])
                    for i in range(len(coef))]) - self.level

    def grad(self, coef):
        coef = self._wrap(coef)
        return self.weight

    def hess(self, coef):
        coef = func._wrap(coef)
        hess = []
        for i in xrange(len(self.weight)):
            hess_line = []
            for j in xrange(i + 1):
                hess_line.append(np.multiply.outer(
                    np.zeros(self.weight[j].shape),
                    np.zeros(self.weight[i].shape)))
            hess.append(hess_line)
        return hess

    def add_feature(self, coef_guess, weight):
        self.weight.append(weight * np.ones(coef_guess.shape))

    def set_level(self, c):
        self.level = c


class poisson:
    def __init__(self, N, X):
        """
        Class to model data under a Poisson Regression.
        """
        self.constraint = []
        self.reg = []

    def fit(self, observations):
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
        df = observations.to_frame('s')
        df['f'] = 1  # frequency
        # sum repeated occurrences for the same index entry
        df = df.groupby(df.index).sum()
        df = df.reindex(pd.MultiIndex.from_product(axis)).fillna(0)
        self.X = df['s'].values.reshape(shape)
        self.N = df['f'].values.reshape(shape)

    def set_coef(self, coef_guess, coerc=None):
        """
        Sets the coefficients.

        Parameters
        ----------
        coef_guess: list
            Initial guess for coefficients. It must be a list of arrays.
        coerc: list (optional)
            List of boolean arrays with value = True if the coefficient has a
            coerced value.
        """
        n = len(coef_guess)
        self.coef = coef_guess
        self.split_ = np.array(
            [0] + [self.coef[i].size for i in xrange(n)]).cumsum()

        # Set coerc arrays if not defined
        if coerc is None:
            coerc = [np.zeros(self.coef[i].shape, np.bool) for i in xrange(n)]
        elif isinstance(coerc, list):
            for i in xrange(n):
                if isinstance(coerc[i], bool):
                    coerc[i] = coerc[i] * np.ones(coef_guess[i].shape, bool)
                else:
                    if isinstance(coerc[i], (list, tuple)):
                        coerc[i] = np.array(coerc[i])

                    if isinstance(coerc[i], np.ndarray):
                        if coerc[i].shape != coef_guess[i].shape:
                            raise ValueError("""The elements of coerc need to
                                have the same shapes as coef elements""")
                    else:
                        raise ValueError("""The elements of coerc have to be a
                            boolean or array_like""")
        else:
            raise ValueError("coerc has to be a list")

        self.free = map(lambda x: ~x, coerc)

        # coerced values
        self.coef_coerc = np.concatenate(
            [coef_guess[i][coerc[i]] for i in xrange(n)])

        # map to insert coerced values
        self.coerc_map = (lambda x: x - np.arange(x.size))((
            lambda x: np.arange(x.size)[x])(np.concatenate(
                [coerc[i].flatten() for i in xrange(n)])))

    def set_model(self, Y_func):
        """
        Sets the regression function.

        Parameters
        ----------
        Y_func : function
            function with argument 'coef'.
        """
        if Y_func(self.coef).shape != self.N.shape:
            raise ValueError("""Y_func needs to return an array with the same
                shape as N and X""")
        self.Y = Y_func

    def set_L(self, grad_L, hess_L):
        """
        Sets the likelihood function gradient and hessian.
        Both grad_L and hess_L need to accept N, X, Y, coef as arguments.

        Parameters
        ----------
        grad_L: function
            function that returns an array of size equal to the number of
            coefficients.
        hess_L: function
            function that returns a triangular matrix with width and height
            equal to the number of coefficients.
        """
        self.grad_L = grad_L
        self.hess_L = hess_L

    def add_constraint(self, coef_map, g):
        """
        Adds a constraint factor to the objective function.

        Parameters
        ----------
        coef_map : int, list
            index of the coefficients to witch g applies
        g : func
            Constraint function.
        """
        self.constraint.append((len(self.constraint), coef_map, 0, g))

    def add_reg(self, coef_map, param, h):
        """
        Adds a regularization factor to the objective function.

        Parameters
        ----------
        coef_map : int, list
            index of the coefficients to witch f applies
        param : float
            Scale parameter to apply to f
        h : func
            Regularization function
        """
        self.reg.append((len(self.reg), coef_map, param, h))

    def E(self):
        """
        Cost / Energy function, to minimize.
        """
        E = self.Y_val.sum() - \
            (np.nan_to_num(self.X * np.ma.log(self.Y_val))).sum()

        for r, coef_map, param, h in self.reg:
            E += param * h.eval(map(lambda k: self.coef[k], coef_map))

        return E

    def Akaine_information_criterion(self):
        """
        Akaike information criterion.
        (The best model is that which minimizes it)
        """
        # k: # of free parameters
        k = sum([c.size for c in self.coef]) - len(self.constraint)

        return 2 * k * (1 + (k - 1) / (self.N.sum() - k - 1)) - 2 * self.E()

    def Baysian_information_criterion(self):
        """
        Bayesian information criterion.
        (The best model is that which minimizes it)
        """
        # k: # of free parameters
        k = sum([c.size for c in self.coef]) - len(self.constraint)

        return k * np.log(self.N.sum()) - 2 * self.E()

    def __reshape_array(self, flat_array, val=0):
        """
        Reshapes as array in order to have the same format as self.coef

        Parameters
        ----------
        flat_array: ndarray
            Array with one axis with the same lenght as the number of free
            coeficients.
        val: float, ndarray (optional)
            Value(s) to fill in coerc_map places. If it is a ndarray it
            needs to have the same size as coerc_map.
        """
        return [np.insert(flat_array, self.coerc_map, val)[
                self.split_[i]:self.split_[i + 1]].
                reshape(self.coef[i].shape)
                for i in xrange(len(self.coef))]

    def __reshape_coef(self, coef_free):
        return self.__reshape_array(coef_free, self.coef_coerc)

    def __reshape_matrix(self, matrix, value=np.NaN):
        if matrix.ndim != 2:
            raise ValueError("'matrix' needs to have two axis")
        elif matrix.shape[0] != matrix.shape[1]:
            raise ValueError("'matrix' needs to be a square")

        n = len(self.coef)

        # Insert columns and rows ith coerced values
        matrix = np.insert(matrix, self.coerc_map, value, axis=0)
        matrix = np.insert(matrix, self.coerc_map, value, axis=1)

        # Split by blocks
        matrix = [[matrix[self.split_[i]:self.split_[i + 1],
                          self.split_[j]:self.split_[j + 1]]
                   for j in xrange(n)] for i in xrange(n)]

        # Reshape each block accordingly
        matrix = [[matrix[i][j].reshape(np.concatenate((
                   self.coef[i].shape, self.coef[j].shape)))
                   for j in xrange(n)] for i in xrange(n)]

        return matrix

    def fisher_matrix(self):
        """
        Observed Information Matrix.
        """
        return self.__reshape_matrix(-self.__flat_hess)

    def error_matrix(self):
        """
        Covariance Matrix.
        """
        return self.__reshape_matrix(-np.linalg.inv(self.__flat_hess))

    def std_error(self):
        """
        Standard Deviation Array.
        """
        return self.__reshape_array(
            np.sqrt(np.diag(-np.linalg.inv(self.__flat_hess))))

    def dispersion(self):
        """
        Returns the normalized dispersion of the observed data.
        Theoretically, it should be 1.
        """
        pass

    def __step(self):

        n = len(self.coef)
        cstr_len = len(self.constraint)

        # --------------------------------------------------------------------
        # 1st phase: Evaluate and sum
        # --------------------------------------------------------------------
        grad = self.grad_L(self.N, self.X, self.Y_val, self.coef) + \
            [0] * len(self.constraint)
        hess = self.hess_L(self.N, self.X, self.Y_val, self.coef)

        # Add blocks corresponding to constraint variables:
        # Hess_lambda_coef = grad_g
        hess_c = [[np.zeros(self.coef[j].shape)
                   for j in xrange(n)] for i in xrange(cstr_len)]

        # --------------------------------------------------------------------
        # 2nd phase: Add constraints / regularization to grad and hess
        # --------------------------------------------------------------------
        for c, coef_map, gamma, g in self.constraint:
            if isinstance(coef_map, list):
                args = map(lambda k: self.coef[k], coef_map)
                grad_g = g.grad(args)
                hess_g = g.hess(args)
                for i in xrange(len(coef_map)):
                    idx = coef_map[i]
                    grad[idx] += grad_g[i]
                    grad[n + c] = np.array([g.eval(args)])
                    hess_c[c][idx] += grad_g[i]
                    for j in xrange(i + 1):
                        hess[idx][coef_map[j]] += hess_g[i][j]

        for r, coef_map, a, h in self.reg:
            if isinstance(coef_map, list):
                args = map(lambda k: self.coef[k], coef_map)
                grad_h = h.grad(args)
                hess_h = h.hess(args)
                for i in xrange(len(coef_map)):
                    idx = coef_map[i]
                    grad[idx] -= a * grad_h[i]
                    for j in xrange(i + 1):
                        hess[idx][coef_map[j]] -= a * hess_h[i][j]

        # --------------------------------------------------------------------
        # 3rd phase: Reshape and flatten
        # --------------------------------------------------------------------
        coef = [self.coef[i][self.free[i]] for i in xrange(n)]
        grad = [grad[i][self.free[i]] for i in xrange(n)] + grad[n:]

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
            for j in xrange(i + 1)] for i in xrange(n)]
        hess = [[hess[i][j].transpose() for j in xrange(i)] +
                [hess[j][i] for j in xrange(i, n)] for i in xrange(n)]
        hess_c = [[hess_c[i][j][self.free[j]] for j in xrange(n)]
                  for i in xrange(cstr_len)]

        # --------------------------------------------------------------------
        # 4th phase: Consolidate blocks
        # --------------------------------------------------------------------
        coef = np.concatenate(coef)
        grad = np.concatenate(grad)
        hess_c = np.vstack(map(np.concatenate, hess_c))
        hess = np.vstack([
            np.hstack([np.vstack(map(np.hstack, hess)), hess_c.transpose()]),
            np.hstack([hess_c, np.zeros((cstr_len, cstr_len))])])

        # --------------------------------------------------------------------
        # 5th phase: Compute
        # --------------------------------------------------------------------
        u = 1
        d = -np.linalg.solve(hess, grad)[:-cstr_len]
        change = np.linalg.norm(d) / np.linalg.norm(coef)
        for i in xrange(5):
            new_coef = self.__reshape_coef(coef + u * d)
            self.Y_val = self.Y(new_coef)
            new_E = self.E()
            if new_E < self.E_val:
                self.coef = new_coef
                self.E_val = new_E
                self.__flat_hess = hess
                return change
            else:
                u *= .75

        self.coef = new_coef
        self.E_val = new_E
        self.__flat_hess = hess
        return change

    def run(self, e=0.001, max_steps=1000):
        """
        Run the algorithm to find the best fit.

        Parameters
        ----------
        e : float (optional)
            Tolerance for termination.
        max_steps : int (optional)
            Maximum number of steps that the algorithm will perform.

        Returns
        -------
        out : boolean
            Returns True if the algorithm converges, otherwise returns
            False.
        """
        self.Y_val = self.Y(self.coef)
        self.E_val = self.E()
        for i in xrange(max_steps):
            print i, self.E_val
            change = self.__step()
            if change < e:
                return True
        return False
