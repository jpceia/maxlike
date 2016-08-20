import numpy as np
import pandas as pd


def series_to_ndarray(s):
    """
    Accepts a pandas.Series with a multiIndex and returns two ndarrays, I and X
    with the same dimensions as the levels of the MultiIndex.
    I refers to the frequency of observations
    X refers to the sum of the observations
    """
    axis = [level.sort_values().values for level in s.index.levels]
    axis_names = list(s.index.names)
    shape = map(len, axis)
    df = s.to_frame('s')
    df['f'] = 1  # frequency
    # sum repeated ocurrences for the same index entry
    df = df.groupby(df.index).sum()
    df = df.reindex(
        pd.MultiIndex.from_product(axis, names=axis_names)).fillna(0)
    X = df['s'].values.reshape(shape)
    I = df['f'].values.reshape(shape)
    return I, X, axis


class poisson:
    def __init__(self, I, X):
        """
        I: observation frequency for a given parameter
        X: sum of observations for a given parameter
        """
        assert I.shape == X.shape, "I and X must have the same shape"
        self.I = I
        self.X = X
        self.constraint = []
        self.reg = []

    def set_coef(self, coef_guess, coerc=None):
        """
        coef_guess: initial guess for coeficients. It must be a list of arrays
        coerc: list of boolean arrays with value = True if the coeficient has a
        coerced value
        """
        N = len(coef_guess)
        self.coef = coef_guess
        self.split_ = np.array(
            [0] + [self.coef[i].size for i in xrange(N)]).cumsum()

        # Set coerc arrays if not defined
        if coerc is None:
            coerc = [np.zeros(self.coef[i].shape, bool)
                     for i in xrange(N)]
        else:
            assert isinstance(coerc, list) and \
                all([coerc[i].shape == coef_guess[i].shape
                     for i in xrange(len(coerc))]), \
                "coerc must be a list of arrays with the same shape as coef"

        self.free = map(lambda x: ~x, coerc)

        # coerced values
        self.coef_coerc = np.concatenate(
            [coef_guess[i][coerc[i]] for i in xrange(N)])

        # index to insert coerced values
        self.coerc_index = (lambda x: x - np.arange(x.size))((
            lambda x: np.arange(x.size)[x])(np.concatenate(
                [coerc[i].flatten() for i in xrange(N)])))

    def set_model(self, Y_func):
        """
        Y_func : function with argument 'coef'
        """
        assert hasattr(self, "coef"), "set_coef must be called first"
        self.Y_func = Y_func
        assert self.Y(self.coef).shape == self.I.shape, \
            "Y_func must return an array with the same shape as I and X"

    def Y(self, coef):
        assert hasattr(self, "Y_func"), "set_model must be called first"
        return self.Y_func(self.I, coef)

    def set_L(self, grad_L, hess_L):
        """
        grad_L must be an array of size equal to the number of coeficients
        hess_L must be a triangular matrix with width and height equal to the
        number of coeficients
        Both must accept I, X, Y, coef as arguments.
        """
        self.grad_L = grad_L
        self.hess_L = hess_L

    def add_constraint(self, index, g, grad_g, hess_g):
        """
        'index' is an index or a list of indexes
         - If 'index' is an index, grad_g and hess_g must return arrays with
         the same shape as coef_index and its square respectively.
         - If index is a list of indexes, grad_g and hess_g must return arrays
         with lengths equal to len(index) and its elements must be arrays with
         shapes equal to coef and coef x coef.
        """
        if isinstance(index, list):
            assert index == sorted(index), \
                "If the index is a list, it must be sorted"

        self.constraint.append(
            (len(self.constraint), index, 0, g, grad_g, hess_g))

    def add_reg(self, index, param, h, grad_h, hess_h):
        """
        'index' is an index or a list of indexes
         - If i is an index, grad_g and hess_g must return arrays with the same
         shape as coef_i and its square respectively.
         - If i is a list of indexes, grad_g and hess_g must return arrays with
         lengths equal to len(i) and its elements must be arrays with shapes
        """
        if isinstance(index, list):
            assert index == sorted(index), \
                "If the index is a list, it must be sorted"

        assert param > 0, "The regularization parameter must be positive"

        self.reg.append(
            (len(self.reg), index, param, h, grad_h, hess_h))

    def E(self):
        """
        Cost function
        """
        assert hasattr(self, "Y_val"), "set_model should be called first"
        E = self.Y_val.sum() - (np.nan_to_num(self.X * np.ma.log(self.Y_val))).sum()

        for r, index, param, h, grad_h, hess_h in self.reg:
            if isinstance(index, list):
                E += param * h(map(lambda k: self.coef[k], index))
            else:
                E += param * h(self.coef[index])
        return E / self.I.sum()

    def __reshape_array(self, flat_array, val=0):
        return [np.insert(flat_array, self.coerc_index, val)[
                self.split_[i]:self.split_[i + 1]].
                reshape(self.coef[i].shape)
                for i in xrange(len(self.coef))]

    def __reshape_coef(self, coef_free):
        return self.__reshape_array(coef_free, self.coef_coerc)

    def __reshape_matrix(self, matrix, value=np.NaN):
        assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1], \
            "'matrix' must be an array with (N, N) shapex"

        N = len(self.coef)

        # insert columns and rows ith coerced values
        matrix = np.insert(matrix, self.coerc_index, value, axis=0)
        matrix = np.insert(matrix, self.coerc_index, value, axis=1)

        # split by blocks
        matrix = [[matrix[self.split_[i]:self.split_[i + 1],
                          self.split_[j]:self.split_[j + 1]]
                   for j in xrange(N)] for i in xrange(N)]

        # reshape each block accordingly
        matrix = [[matrix[i][j].reshape(np.concatenate((
                   self.coef[i].shape, self.coef[j].shape)))
                   for j in xrange(N)] for i in xrange(N)]

        return matrix

    def fisher_matrix(self):
        """
        Observed Information Matrix
        """
        assert hasattr(self, '__flat_hess'), "step method must be called first"
        return self.__reshape_matrix(-self.__flat_hess)

    def error_matrix(self):
        """
        Covariance Matrix
        """
        assert hasattr(self, '__flat_hess'), "step method must be called first"
        return self.__reshape_matrix(-np.linalg.inv(self.__flat_hess))

    def std_error(self):
        """
        Standard Deviation array
        """
        # assert hasattr(self, '__flat_hess'), "step method must be called first"
        return self.__reshape_array(
            np.sqrt(np.diag(-np.linalg.inv(self.__flat_hess))))

    def __step(self):
        """
        """

        N = len(self.coef)
        M = len(self.constraint)

        # --------------------------------------------------------------------
        # 1st phase: Evaluate and sum
        # --------------------------------------------------------------------
        grad = self.grad_L(self.I, self.X, self.Y_val, self.coef) + \
            [0] * len(self.constraint)
        hess = self.hess_L(self.I, self.X, self.Y_val, self.coef)

        # Add blocks corresponding to constraint variables:
        # Hess_lambda_coef = grad_g
        hess_c = [[np.zeros(self.coef[j].shape)
                   for j in xrange(N)] for i in xrange(M)]

        # --------------------------------------------------------------------
        # 2nd phase: Add constraints / regularization to grad and hess
        # --------------------------------------------------------------------
        for c, index, gamma, g, grad_g, hess_g in self.constraint:
            if isinstance(index, list):
                args = map(lambda k: self.coef[k], index)  # args = coef[index]
                grad_g_val = grad_g(args)
                hess_g_val = hess_g(args)
                for i in xrange(len(index)):
                    I = index[i]
                    grad[I] += grad_g_val[i]
                    grad[N + c] = np.array([g(args)])
                    hess_c[c][I] += grad_g_val[i]
                    for j in xrange(i + 1):
                        J = index[j]
                        hess[I][J] += hess_g_val[i][j]
            else:
                k = index
                grad_g_val = grad_g(self.coef[k])
                grad[k] += gamma * grad_g_val
                grad[N + c] = np.array([g(self.coef[k])])
                hess[k][k] += gamma * hess_g(self.coef[k])
                hess_c[c][k] += grad_g_val

        for r, index, a, h, grad_h, hess_h in self.reg:
            if isinstance(index, list):
                args = map(lambda k: self.coef[k], index)  # args = coef[index]
                grad_h_val = grad_h(args)
                hess_h_val = hess_h(args)
                for i in xrange(len(index)):
                    I = index[i]
                    grad[I] -= a * grad_h_val[i]
                    for j in xrange(i + 1):
                        J = index[j]
                        hess[I][J] -= a * hess_h_val[i][j]
            else:
                k = index
                grad[k] -= a * grad_h(self.coef[k])
                hess[k][k] -= a * hess_h(self.coef[k])

        # --------------------------------------------------------------------
        # 3rd phase: Reshape and flatten
        # --------------------------------------------------------------------
        coef = [self.coef[i][self.free[i]] for i in xrange(N)]
        grad = [grad[i][self.free[i]] for i in xrange(N)] + grad[N:]

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
            for j in xrange(i + 1)] for i in xrange(N)]
        hess = [[hess[i][j].transpose() for j in xrange(i)] +
                [hess[j][i] for j in xrange(i, N)]
                for i in xrange(N)]
        hess_c = [[hess_c[i][j][self.free[j]] for j in xrange(N)]
                  for i in xrange(M)]

        # --------------------------------------------------------------------
        # 4th phase: Consolidate blocks
        # --------------------------------------------------------------------
        coef = np.concatenate(coef)
        grad = np.concatenate(grad)
        hess_c = np.vstack(map(np.concatenate, hess_c))
        hess = np.vstack([
            np.hstack([np.vstack(map(np.hstack, hess)), hess_c.transpose()]),
            np.hstack([hess_c, np.zeros((M, M))])])

        # --------------------------------------------------------------------
        # 5th phase: Compute
        # --------------------------------------------------------------------
        u = 1
        d = -np.linalg.solve(hess, grad)[:-M]
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
        Run the algorithm to find the best fit
        Parameters:
        - e : the algorithm stops when |dx|/|x| < e
        - max_steps : the algorithm stops, and fails, if the max_steps are
          performed without having |dx|/|x| < e before.
        """
        self.Y_val = self.Y(self.coef)
        self.E_val = self.E()
        for i in xrange(max_steps):
            print i, self.E_val
            change = self.__step()
            if change < e:
                return True
        print "Error, the process didn't converge"
        return False
