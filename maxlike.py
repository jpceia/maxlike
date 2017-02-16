import numpy as np
import pandas as pd


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
        self.factors = []
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

    def add_factor(self, param_map, feat_map, new_factor, weight=1):
        """
        Adds a factor to the model.

        Parameters
        ----------
        param_map: int, list
            index of parameters that 'new_factor' accepts
        feat_map:
            index of features that 'new_factor' accepts
        new_factor: func
            factor to add to the model
        """
        if isinstance(param_map, int):
            param_map = [param_map]
        if isinstance(feat_map, int):
            feat_map = [feat_map]

        self.factors.append((param_map, feat_map, new_factor, weight))

    def __slice(self, feat_map):
        """
        Returns a slice that induces broadcast accross the dimentions that are
        not in feat_map
        """
        return map(
            lambda x: slice(None) if x else None,
            (np.arange(self.N.ndim)[:, None] ==
                np.array(feat_map)[None, :]).all(1))

    def ln_y(self, params):
        """
        Calculates the logarithm of the model
        """
        F = np.zeros(self.N.shape)
        for param_map, feat_map, f, weight in self.factors:
            F += weight * \
                f.eval(map(params.__getitem__,
                           feat_map))[self.__slice(feat_map)]
        return F

    def G(self, params):
        """
        Likelihood function, to maximize.
        """
        ln_y = self.ln_y(params)
        G = (self.X * ln_y - self.N * np.exp(ln_y)).sum()
        for r, param_map, gamma, h in self.reg:
            G += gamma * h.eval(map(lambda k: params[k], param_map))

        return G

    def __grad_ln_y(self, params):
        """
        Calculates the gradient of the model
        """
        grad = [np.zeros(self.N.shape + p.shape) for p in self.param]

        for param_map, feat_map, f, weight in self.factors:
            # since der[k] is going to have shape N.shape x param[k].shape
            # we need to construct an appropriate slice to expand f.grade()
            # to have aligned dimentions
            slc = self.__slice(feat_map) + [Ellipsis]

            # we copy the value of f.grad since we're going to use it
            # several times.
            grad_f = f.grad(map(params.__getitem__, feat_map))
            for i in param_map:
                grad[i] += weight * grad_f[slc]

        return grad

    def grad_L(self, params):
        """
        Calculates the gradient of the log-likelihood function.
        """
        delta = self.X - self.N * np.exp(self.ln_y(params))
        der = self.__grad_ln_y(params)
        return [(delta[[Ellipsis] + [None] * self.param[i].ndim] *
                 der[i]).sum(tuple(range(self.N.ndim)))
                for i in range(len(self.param))]

    def hess_L(self, params):
        """
        Calculates the hessian of the log-likelihood function.
        """
        hess = []
        ln_y = self.ln_y(params)
        delta = self.X - self.N * np.exp(ln_y)
        grad = self.__grad_ln_y(params)
        slc = []
        slc_expd = []
        slc_feat = [slice(None)] * self.N.ndim

        for i in range(len(self.param)):
            slc.append([slice(None)] * self.param[i].ndim)
            slc_expd.append([None] * self.param[i].ndim)
            hess_line = []
            for j in range(i + 1):
                cube = np.zeros(self.N.shape +
                                self.param[j].shape +
                                self.param[i].shape)
                for param_map, feat_map, f, weight in self.factors:
                    if i in param_map and j in param_map:
                        feat_expd = self.__slice(feat_map)
                        cube += weight * \
                            f.hess(map(params.__getitem__, feat_map),
                                   i, j)[feat_expd + slc[j] + slc[i]]
                h = -((self.N * np.exp(ln_y)
                       )[slc_feat + slc_expd[j] + slc_expd[i]] *
                      grad[j][slc_feat + slc[j] + slc_expd[i]] *
                      grad[i][slc_feat + slc_expd[j] + slc[i]]
                      ).sum(tuple(np.arange(self.N.ndim)))
                h += (delta[slc_feat + slc_expd[j] + slc_expd[i]] * cube).sum(
                    tuple(np.arange(self.N.ndim)))
                hess_line.append(h)
            hess.append(hess_line)
        return hess

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
        self.reg.append((len(self.reg), param_map, gamma, h))

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

    def __step(self, max_steps=5):

        n = len(self.param)
        cstr_len = len(self.constraint)

        # --------------------------------------------------------------------
        # 1st phase: Evaluate and sum
        # --------------------------------------------------------------------
        grad = self.grad_L(self.param) + \
            [0] * len(self.constraint)
        hess = self.hess_L(self.param)

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

        for r, param_map, gamma, h in self.reg:
            if isinstance(param_map, list):
                args = map(lambda k: self.param[k], param_map)
                grad_h = h.grad(args)
                hess_h = h.hess(args)
                for i in xrange(len(param_map)):
                    idx = param_map[i]
                    grad[idx] -= gamma * grad_h[i]
                    for j in xrange(i + 1):
                        hess[idx][param_map[j]] -= gamma * hess_h[i][j]

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
        for i in xrange(max_steps):
            new_param = self.__reshape_param(param + u * d)
            new_G = self.G(new_param)
            if new_G > self.G_val:
                self.param = new_param
                self.G_val = new_G
                self.__flat_hess = hess  # ??? why here and not before
                return change
            else:
                u *= .5

        print """Error: the objective function did non increased after %d
                 steps""" % max_steps

    def run(self, tol=1e-3, max_steps=100):
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
        self.G_val = self.G(self.param)
        for i in xrange(max_steps):
            print i, self.G_val
            change = self.__step()
            if change < tol:
                return True
        return False
