import abc
import numpy as np
from random import getrandbits
from .tensor import Tensor


class ConvergenceError(Exception):
    def __init__(self, message, type=0):
        super(ConvergenceError, self).__init__(message)
        self.type = type


class Param(np.ma.MaskedArray):
    def __new__(cls, data, *args, **kwargs):
        obj = super(Param, cls).__new__(cls, data, *args, **kwargs)
        obj.hash = getrandbits(128)
        return obj

    def __hash__(self):
        return self.hash


class MaxLike(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
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

    def reset_params(self):
        self.params = ()

    def g(self, params):
        """
        Objective function, to maximize.
        """
        g = self.like(params)
        for param_map, h in self.reg:
            g -= h([params[k] for k in param_map])
        res = g / self.N.sum()
        return float(res.values)

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
        try:
            self.params = self.params + (Param(values, mask=fixed), )
        except AttributeError:
            self.params = ()
            self.add_param(values, fixed)

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
        if isinstance(param_map, int):
            param_map = [param_map]

        self.constraint.append((param_map, 0, g))

    def add_regularization(self, param_map, h):
        """
        Adds a regularization factor to the objective function.

        Parameters
        ----------
        param_map : int, list
            index of the parameters to witch f applies
        h : func
            Regularization function
        """
        if isinstance(param_map, int):
            param_map = [param_map]

        self.reg.append((param_map, h))

    def akaine_information_criterion(self):
        """
        Akaike information criterion
        """
        # {free parameters}
        k = sum(map(np.ma.count, self.params)) - len(self.constraint)
        return 2 * k * (1 + (k - 1) / (self.N.sum() - k - 1)) - \
            2 * self.g(self.params)

    def bayesian_information_criterion(self):
        """
        Bayesian information criterion
        """

        # {free parameters}
        k = sum(map(np.ma.count, self.params)) - len(self.constraint)
        return k * np.log(self.N.sum()) - 2 * self.g(self.params)

    def _sum_feat(self):
        return tuple(-np.arange(self.N.ndim) - 1)

    def _reshape_array(self, flat_array, val=np.nan):
        """
        Reshapes as array in order to have the same format as self.params

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
            for p in self.params:
                s_1 = s_0 + p.count()
                data = np.insert(
                    flat_array[s_0:s_1],
                    fix_map(p.mask.flatten()),
                    val).reshape(p.shape)
                shaped_array.append(Param(data, mask=p.mask))
                s_0 = s_1
        # val is an array
        else:
            f_0 = 0
            for p in self.params:
                s_1 = s_0 + p.count()
                f_1 = f_0 + p.mask.sum()
                data = np.insert(
                    flat_array[s_0:s_1],
                    fix_map(p.mask.flatten()),
                    val[f_0:f_1]).reshape(p.shape)
                shaped_array.append(Param(data, mask=p.mask))
                s_0 = s_1
                f_0 = f_1
        return tuple(shaped_array)

    def _reshape_params(self, params_free):
        return self._reshape_array(
            params_free,
            np.concatenate([p[p.mask].data for p in self.params]))

    def _reshape_matrix(self, matrix, val=np.nan):
        if matrix.ndim != 2:
            raise ValueError("matrix.ndim != 2")
        elif matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix.shape[0] != matrix.shape[1]")

        # Split by blocks
        s_ = [0]
        f_ = [0]
        val_map = []
        for p in self.params:
            s_.append(s_[-1] + p.count())
            f_.append(f_[-1] + p.mask.sum())
            val_map.append((lambda x: x - np.arange(x.size))(
                (lambda x: np.arange(x.size)[x])(p.mask.flatten())))

        # val is a scalar
        if isinstance(val, (int, float)):
            return [[np.insert(np.insert(
                matrix[s_[i]:s_[i + 1], s_[j]:s_[j + 1]],
                val_map[i], val, 0),
                val_map[j], val, 1).reshape(
                p_i.shape + p_j.shape)
                for j, p_j in enumerate(self.params)]
                for i, p_i in enumerate(self.params)]
        else:
            return [[np.insert(np.insert(
                matrix[s_[i]:s_[i + 1], s_[j]:s_[j + 1]],
                val_map[i], val[f_[i]:f_[i + 1]], 0),
                val_map[j], val[f_[j]:f_[j + 1]], 1).reshape(
                p_i.shape + p_j.shape)
                for j, p_j in enumerate(self.params)]
                for i, p_i in enumerate(self.params)]

    def fisher_matrix(self):
        return self._reshape_matrix(self.flat_hess_, 0)

    def error_matrix(self):
        """
        Covariance Matrix.
        """
        return self._reshape_matrix(-np.linalg.inv(self.flat_hess_))

    def std_error(self):
        """
        Standard Deviation Array.
        """
        cut = -len(self.constraint)
        if cut == 0:
            cut = None
        return self._reshape_array(np.sqrt(np.diag(-np.linalg.inv(
            self.flat_hess_))[:cut]))

    def __step(self, verbose=False):
        max_steps = 10
        n = len(self.params)
        c_len = len(self.constraint)

        # --------------------------------------------------------------------
        # 1st phase: Evaluate and sum
        # --------------------------------------------------------------------
        grad = self.grad_like(self.params) + [0] * c_len
        hess = self.hess_like(self.params)

        # Add blocks corresponding to constraint variables:
        # Hess_lambda_params = grad_g
        hess_c = [[np.zeros_like(p)
                   for p in self.params]
                  for _ in range(c_len)]

        # --------------------------------------------------------------------
        # 2nd phase: Add constraints / regularization to grad and hess
        # --------------------------------------------------------------------
        for k, (param_map, gamma, g) in enumerate(self.constraint):
            args = [self.params[k] for k in param_map]
            grad_g = g.grad(args)
            hess_g = g.hess(args)
            for i, idx in enumerate(param_map):
                grad[idx] += gamma * grad_g[i]
                grad[n + k] = np.asarray([g(args)])
                hess_c[k][idx] += np.array(grad_g[i])
                for j in range(i + 1):
                    hess[idx][param_map[j]] += gamma * hess_g[i][j]

        for param_map, h in self.reg:
            args = [self.params[k] for k in param_map]
            grad_h = h.grad(args)
            hess_h = h.hess(args)
            for i, idx in enumerate(param_map):
                grad[idx] -= grad_h[i]
                for j in range(i + 1):
                    hess[idx][param_map[j]] -= hess_h[i][j]

        # --------------------------------------------------------------------
        # 3rd phase: Reshape and flatten
        # --------------------------------------------------------------------
        flat_params = [p.compressed() for p in self.params]
        grad = [g.values[~p.mask] for g, p in zip(grad, self.params)] + \
                grad[n:]

        # ------
        # | aa |
        # -----------
        # | ab | bb |
        # ----------------
        # | ac | bc | cc |
        # ----------------
        #
        # then hess[i][j].shape = shape[j] x shape[i]

        hess = [[hess[j][i].values[np.multiply.outer(
            ~self.params[i].mask, ~p_j.mask)].reshape(
            (self.params[i].count(), p_j.count()))
            for i in range(j + 1)] for j, p_j in enumerate(self.params)]
        hess = [[hess[i][j].transpose() for j in range(i)] +
                [hess[j][i] for j in range(i, n)] for i in range(n)]
        hess_c = [[hess_c[i][j][~p_j.mask]
                   for j, p_j in enumerate(self.params)]
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
        d = np.linalg.solve(hess, grad)
        if c_len > 0:
            d = d[:-c_len]

        # change = np.linalg.norm(d) / np.linalg.norm(params)
        for i in range(max_steps):
            new_params = self._reshape_params(flat_params - u * d)
            new_g = self.g(new_params)
            if new_g - self.g_last >= 0:
                self.params = new_params
                self.g_last = new_g
                self.flat_hess_ = hess
                return None
            else:
                u *= .5
                if verbose:
                    print("\tu=", u)

        raise ConvergenceError(
            "Error: the objective function did not increase " +
            "after %d steps" % max_steps, "step")

    def fit(self, tol=1e-8,  max_steps=None, verbose=0, scipy=0, **kwargs):
        """
        Run the algorithm to find the best fit.

        Parameters
        ----------
        tol : float (optional)
            Tolerance for termination.

        max_steps : int (optional)
            Maximum number of steps that the algorithm will perform.
            Default =50 when scipy option is enabled, otherwise =20.

        verbose : int (optional)
            Sets the verbosity of the algorithm.
            0 : no messages (default)
            1 : low verbosity
            2 : high verbosity

        scipy: int (optional)
            0 : scipy is not used (default)
            1 : scipy using func values
            2 : scipy using func and grad values 

        Returns
        -------
        out : boolean
            Returns True if the algorithm converges, otherwise returns
            False.
        """
        dim = getattr(self, 'dim', 0)
        for k, v in kwargs.items():
            self.__dict__[k] = Tensor(v, dim=dim)

        if scipy > 0:
            if max_steps is None:
                max_steps = 50  # default value for SciPy

            from scipy.optimize import minimize

            use_jac = scipy > 1  # SciPy==2 to use gradient for optimization

            if use_jac:
                def opt_like(_flat_params):
                    params = self._reshape_params(_flat_params)
                    jac = self.grad_like(params)
                    flat_jac = np.concatenate([
                        j.values[~p.mask] for j, p in zip(jac, self.params)])
                    return -self.g(params), -flat_jac / self.N.sum().values
            else:
                def opt_like(_flat_params):
                    params = self._reshape_params(_flat_params)
                    return -self.g(params)

            constraints = []
            for param_map, _, g in self.constraint:
                def foo_constraint(_flat_params):
                    params = self._reshape_params(_flat_params)
                    return g([params[k] for k in param_map])
                constraints.append({
                    'type': 'eq',
                    'fun': foo_constraint})

            flat_params = np.concatenate(
                [p.compressed() for p in self.params])

            res = minimize(
                opt_like, flat_params,
                method="SLSQP",
                jac=use_jac,
                tol=tol,
                constraints=constraints,
                options={
                    'maxiter': max_steps,
                    'disp': verbose > 0,
                    'ftol': tol,
                    'iprint': verbose,
                })

            self.params = self._reshape_params(res.x)
            self.g_last = -res.fun
            if not res.success:
                raise ConvergenceError(res.message)
        else:
            if max_steps is None:
                max_steps = 20  # default value for custom model

            self.g_last = self.g(self.params)
            for i in range(max_steps):
                old_g = self.g_last
                self.__step(verbose > 1)
                if verbose > 0:
                    print(i, self.g_last)
                if abs(old_g / self.g_last - 1) < tol:
                    return None
            raise ConvergenceError("Error: the objective function did not",
                                   "converge after %d steps" % max_steps)
