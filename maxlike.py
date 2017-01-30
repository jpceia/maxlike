import numpy as np
import pandas as pd


class func:
    def __init__(self):
        self.params = []

    def eval(self, param, *args, **kargs):
        raise NotImplementedError

    def grad(self, param, *args, **kargs):
        raise NotImplementedError

    def hess(self, param, *args, **kargs):
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

    def eval(self, param):
        param = self._wrap(param)
        return sum([sum(self.weight[i] * param[i])
                    for i in range(len(param))]) - self.level

    def grad(self, param):
        param = self._wrap(param)
        return self.weight

    def hess(self, param):
        param = func._wrap(param)
        hess = []
        for i in xrange(len(self.weight)):
            hess_line = []
            for j in xrange(i + 1):
                hess_line.append(np.multiply.outer(
                    np.zeros(self.weight[j].shape),
                    np.zeros(self.weight[i].shape)))
            hess.append(hess_line)
        return hess

    def add_feature(self, shape, weight):
        self.weight.append(weight * np.ones(shape))

    def set_level(self, c):
        self.level = c


class quadratic(func):
    def __init__(self):
        self.weight = []
        self.level = 0

    def eval(self, param):
        pass

    def grad(self, param):
        pass

    def hess(self, param):
        pass

    def add_feature(self, shape, weight):
        if isinstance(weight, (int, float)):
            weight *= np.ones(shape)
        pass


class onehot(func):
    def __init__(self, lenght):
        self.lenght = lenght
        pass

    def eval(self, param):
        return param

    def grad(self, param):
        pass

    def hess(self, param):
        pass


class dummyLinear(func):
    def __init__(self, lenght):
        self.lenght = lenght

    def eval(self, param):
        return param * np.arange(self.lenght)

    def grad(self, param):
        return np.arange(self.lenght)

    def hess(self, param):
        return np.zeros((self.lenght, self.lenght))


class poisson:
    """
    Class to model data under a Poisson Regression.
    """

    def __init__(self):
        self.param = []
        self.free = []
        self.label = []
        self.split_ = [0]
        self.fixed_map = np.array([], np.int)
        self.param_fixed = np.array([])
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
        df = observations.to_frame('X')
        df['N'] = 1  # frequency
        # sum repeated occurrences for the same index entry
        df = df.groupby(df.index).sum()
        df = df.reindex(pd.MultiIndex.from_product(axis)).fillna(0)
        self.X = df['X'].values.reshape(shape)
        self.N = df['N'].values.reshape(shape)
        return axis

    def add_param(self, param_guess, fixed=None, label=''):
        """
        Adds a new parameter.

        Parameters
        ----------
        param_guess: list
            Initial guess for the parameter.
        fixed: list (optional)
            Boolean arrays with value = True if the parameter has a
            fixed value.
        label: string (optional)
            label for that parameter.
        """
        if isinstance(param_guess, (int, float, tuple, list)):
            param_guess = np.array(param_guess)

        shape = param_guess.shape
        free_size = sum([el.size for el in self.free])
        self.param.append(param_guess)
        self.label.append(label)
        self.split_.append(self.split_[-1] + param_guess.size)

        if fixed is None:
            fixed = np.zeros(shape, np.bool)
        elif isinstance(fixed, bool):
            fixed *= np.ones(shape, np.bool)
        elif isinstance(fixed, np.ndarray):
            if not fixed.shape == shape:
                raise ValueError("""If fixed is a ndarray, it needs to have
                    the same shapes as param elements""")
        else:
            raise ValueError("fixed must be None, a boolean or a ndarray")

        # fixed values
        self.param_fixed = np.concatenate((
            self.param_fixed,
            param_guess[fixed]))

        # map to insert fixed values
        self.free.append(~fixed)
        self.fixed_map = np.concatenate((
            self.fixed_map,
            (lambda x: x - np.arange(x.size))(
                (lambda x: np.arange(x.size)[x])(
                    fixed.flatten())) + free_size
        ))

    def set_model(self, Y_func):
        """
        Sets the regression function.

        Parameters
        ----------
        Y_func : function
            function with argument 'param'.
        """
        if Y_func(self.param).shape != self.N.shape:
            raise ValueError("""Y_func needs to return an array with the same
                shape as N and X""")
        self.Y = Y_func

    def set_L(self, grad_L, hess_L):
        """
        Sets the likelihood function gradient and hessian.
        Both grad_L and hess_L need to accept N, X, Y, param as arguments.

        Parameters
        ----------
        grad_L: function
            function that returns an array of size equal to the number of
            parameters.
        hess_L: function
            function that returns a triangular matrix with width and height
            equal to the number of parameters.
        """
        self.grad_L = grad_L
        self.hess_L = hess_L

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
        self.constraint.append((len(self.constraint), param_map, 0, g))

    def add_reg(self, param_map, param, h):
        """
        Adds a regularization factor to the objective function.

        Parameters
        ----------
        param_map : int, list
            index of the parameters to witch f applies
        param : float
            Scale parameter to apply to f
        h : func
            Regularization function
        """
        self.reg.append((len(self.reg), param_map, param, h))

    def E(self):
        """
        Cost / Energy function, to minimize.
        """
        E = self.Y_val.sum() - \
            (np.nan_to_num(self.X * np.ma.log(self.Y_val))).sum()

        for r, param_map, param, h in self.reg:
            E += param * h.eval(map(lambda k: self.param[k], param_map))

        return E

    def Akaine_information_criterion(self):
        """
        Akaike information criterion.
        (The best model is that which minimizes it)
        """
        # k: # of free parameters
        k = sum([c.size for c in self.param]) - len(self.constraint)

        return 2 * k * (1 + (k - 1) / (self.N.sum() - k - 1)) - 2 * self.E()

    def Baysian_information_criterion(self):
        """
        Bayesian information criterion.
        (The best model is that which minimizes it)
        """
        # k: # of free parameters
        k = sum([c.size for c in self.param]) - len(self.constraint)

        return k * np.log(self.N.sum()) - 2 * self.E()

    def __reshape_array(self, flat_array, val=0):
        """
        Reshapes as array in order to have the same format as self.param

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
                reshape(self.param[i].shape)
                for i in xrange(len(self.param))]

    def __reshape_param(self, param_free):
        return self.__reshape_array(param_free, self.param_fixed)

    def __reshape_matrix(self, matrix, value=np.NaN):
        if matrix.ndim != 2:
            raise ValueError("'matrix' needs to have two axis")
        elif matrix.shape[0] != matrix.shape[1]:
            raise ValueError("'matrix' needs to be a square")

        n = len(self.param)

        # Insert columns and rows ith fixed values
        matrix = np.insert(matrix, self.fixed_map, value, axis=0)
        matrix = np.insert(matrix, self.fixed_map, value, axis=1)

        # Split by blocks
        matrix = [[matrix[self.split_[i]:self.split_[i + 1],
                          self.split_[j]:self.split_[j + 1]]
                   for j in xrange(n)] for i in xrange(n)]

        # Reshape each block accordingly
        matrix = [[matrix[i][j].reshape(np.concatenate((
                   self.param[i].shape, self.param[j].shape)))
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

        n = len(self.param)
        cstr_len = len(self.constraint)

        # --------------------------------------------------------------------
        # 1st phase: Evaluate and sum
        # --------------------------------------------------------------------
        grad = self.grad_L(self.N, self.X, self.Y_val, self.param) + \
            [0] * len(self.constraint)
        hess = self.hess_L(self.N, self.X, self.Y_val, self.param)

        # Add blocks corresponding to constraint variables:
        # Hess_lambda_param = grad_g
        hess_c = [[np.zeros(self.param[j].shape)
                   for j in xrange(n)] for i in xrange(cstr_len)]

        # --------------------------------------------------------------------
        # 2nd phase: Add constraints / regularization to grad and hess
        # --------------------------------------------------------------------
        for c, param_map, gamma, g in self.constraint:
            if isinstance(param_map, list):
                args = map(lambda k: self.param[k], param_map)
                grad_g = g.grad(args)
                hess_g = g.hess(args)
                for i in xrange(len(param_map)):
                    idx = param_map[i]
                    grad[idx] += grad_g[i]
                    grad[n + c] = np.array([g.eval(args)])
                    hess_c[c][idx] += grad_g[i]
                    for j in xrange(i + 1):
                        hess[idx][param_map[j]] += hess_g[i][j]

        for r, param_map, a, h in self.reg:
            if isinstance(param_map, list):
                args = map(lambda k: self.param[k], param_map)
                grad_h = h.grad(args)
                hess_h = h.hess(args)
                for i in xrange(len(param_map)):
                    idx = param_map[i]
                    grad[idx] -= a * grad_h[i]
                    for j in xrange(i + 1):
                        hess[idx][param_map[j]] -= a * hess_h[i][j]

        # --------------------------------------------------------------------
        # 3rd phase: Reshape and flatten
        # --------------------------------------------------------------------
        param = [self.param[i][self.free[i]] for i in xrange(n)]
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
        param = np.concatenate(param)
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
        change = np.linalg.norm(d) / np.linalg.norm(param)
        for i in xrange(5):
            new_param = self.__reshape_param(param + u * d)
            self.Y_val = self.Y(new_param)
            new_E = self.E()
            if new_E < self.E_val:
                self.param = new_param
                self.E_val = new_E
                self.__flat_hess = hess
                return change
            else:
                u *= .75

        self.param = new_param
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
        self.Y_val = self.Y(self.param)
        self.E_val = self.E()
        for i in xrange(max_steps):
            print i, self.E_val
            change = self.__step()
            if change < e:
                return True
        return False
