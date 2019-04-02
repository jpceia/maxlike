import abc
import numpy as np
from functools import wraps
from collections import OrderedDict
from scipy.linalg import pinvh
from random import getrandbits
from six import with_metaclass
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


class MaxlikeBase(type):
    @staticmethod
    def wrapped_like(foo):
        @wraps(foo)
        def wrapper(obj, params):
            return foo(obj, params, **obj.feat_dico)
        return wrapper

    def __new__(cls, name, bases, attrs, **kwargs):
        for f_name in ['g', 'like', 'grad_like', 'hess_like']:
            if f_name in attrs:
                attrs[f_name] = MaxlikeBase.wrapped_like(attrs[f_name])
        return type.__new__(cls, name, bases, attrs, **kwargs)


class MaxLike(with_metaclass(MaxlikeBase, object)):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        # model related
        self.model = None

        self.constraint = []
        self.reg = []

        self.feat_dico = OrderedDict()
        self.feat_dico['N'] = None
        self.g_last = None
        self.converged = False

    @abc.abstractmethod
    def like(self, params, **kwargs):
        """
        Likelihood function.
        """
        pass

    @abc.abstractmethod
    def grad_like(self, params, **kwargs):
        """
        Calculates the gradient of the log-likelihood function.
        """
        pass

    @abc.abstractmethod
    def hess_like(self, params, **kwargs):
        """
        Calculates the hessian of the log-likelihood function.
        """
        pass

    def reset_params(self):
        self.params = ()

    def g(self, params, N, **kwargs):
        """
        Objective function, to maximize.
        """
        g = self.like(params)
        for param_map, h in self.reg:
            g -= h.eval([params[k] for k in param_map])
        res = g / N.sum()
        return float(res)

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
        try:
            mask = self.flat_hess_.mask
        except AttributeError:
            mask = None
        return self._reshape_matrix(
            -np.ma.array(pinvh(self.flat_hess_.data), mask=mask))

    def std_error(self):
        """
        Standard Deviation Array.
        """
        cut = len(self.constraint)
        if cut == 0:
            cut = None
        return self._reshape_array(np.sqrt(np.diag(-np.linalg.inv(
            self.flat_hess_))[:-cut]))

    def flat_grad(self, params):

        n = len(params)
        grad = self.grad_like(params) + [0] * len(self.constraint)

        for k, (param_map, gamma, g) in enumerate(self.constraint):
            args = [params[k] for k in param_map]
            grad_g = g.grad(args)
            for i, idx in enumerate(param_map):
                grad[idx] += gamma * grad_g[i]
                grad[n + k] = [g.eval(args)]

        for param_map, h in self.reg:
            args = [params[k] for k in param_map]
            grad_h = h.grad(args)
            for i, idx in enumerate(param_map):
                grad[idx] -= grad_h[i]

        grad = [g.values[~p.mask] for g, p in zip(grad, params)] + grad[n:]
        grad = np.concatenate(grad)

        return grad

    def flat_hess(self, params):
        n = len(params)
        c_len = len(self.constraint)

        # --------------------------------------------------------------------
        # 1st phase: Evaluate and sum
        # --------------------------------------------------------------------
        hess = self.hess_like(params)

        # Add blocks corresponding to constraint variables:
        # Hess_lambda_params = grad_g
        hess_c = [[np.zeros_like(p) for p in params] for _ in range(c_len)]

        # --------------------------------------------------------------------
        # 2nd phase: Add constraints / regularization to grad and hess
        # --------------------------------------------------------------------
        for k, (param_map, gamma, g) in enumerate(self.constraint):
            args = [params[k] for k in param_map]
            grad_g = g.grad(args)
            hess_g = g.hess(args)
            for i, idx in enumerate(param_map):
                hess_c[k][idx] += grad_g[i]
                for j in range(i + 1):
                    hess[idx][param_map[j]] += gamma * hess_g[i][j]

        for param_map, h in self.reg:
            args = [params[k] for k in param_map]
            hess_h = h.hess(args)
            for i, idx in enumerate(param_map):
                for j in range(i + 1):
                    hess[idx][param_map[j]] -= hess_h[i][j]

        # --------------------------------------------------------------------
        # 3rd phase: Reshape and flatten
        # --------------------------------------------------------------------

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
            ~params[i].mask, ~p_j.mask)].reshape(
            (params[i].count(), p_j.count()))
            for i in range(j + 1)] for j, p_j in enumerate(params)]
        hess = [[hess[i][j].transpose() for j in range(i)] +
                [hess[j][i] for j in range(i, n)] for i in range(n)]
        hess_c = [[hess_c[i][j][~p_j.mask] for j, p_j in enumerate(params)]
                  for i in range(c_len)]

        # --------------------------------------------------------------------
        # 4th phase: Consolidate blocks
        # --------------------------------------------------------------------
        hess = np.vstack(list(map(np.hstack, hess)))

        if c_len > 0:
            hess_c = np.vstack(list(map(np.concatenate, hess_c)))
            hess = np.vstack([
                np.hstack([hess, hess_c.transpose()]),
                np.hstack([hess_c,
                           np.zeros((c_len, c_len))])])

        return hess
     
    def newton_step(self, params):
        grad = self.flat_grad(params)
        hess = self.flat_hess(params)
        step = np.linalg.solve(hess, grad)
        c_len = len(self.constraint)
        step = step[:(-(c_len + 1) % step.shape[0]) + 1]
        return step, grad, hess

    def fit(self, tol=1e-8, max_steps=None, verbose=0, scipy=0, 
            method=None, **kwargs):
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

        method: string (optional)
            Method to calibrate the model
            * newton
            * broyden
            * SLSQP

        Returns
        -------
        out : boolean
            Returns True if the algorithm converges, otherwise returns
            False.
        """
        dim = getattr(self, 'dim', 0)
        for k in self.feat_dico:
            assert k in kwargs
            self.feat_dico[k] = Tensor(kwargs[k], dim=dim)

        if scipy > 0:
            if max_steps is None:
                max_steps = 50  # default value for SciPy

            if method is None:
                method = "SLSQP"

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
                method=method,
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

            return True

        else:
            if max_steps is None:
                max_steps = 20  # default value

            if method is None:
                method = "newton"
            else:
                method = method.lower()

            params = self.params
            flat_params = np.concatenate([p.compressed() for p in params])
            new_g = self.g(params)

            if verbose > 0:
                print(0, new_g)

            if method == "newton":

                max_inner_steps = 5
                for k in range(max_steps):
                    prev_g = new_g
                    step, grad, hess = self.newton_step(params)

                    mult = 1
                    for _ in range(max_inner_steps):
                        new_flat_params = flat_params - mult * step
                        params = self._reshape_params(new_flat_params)
                        new_g = self.g(params)
                        if new_g - prev_g >= 0:
                            flat_params = new_flat_params
                            break
                        else:
                            mult *= .5
                            if verbose > 1:
                                print("\tmultiplier = ", mult)
                    else:
                        self.converged = False
                        raise ConvergenceError(
                            "Error: the objective function did not increase " +
                            "after %d steps" % max_inner_steps, "step")

                    if verbose > 0:
                        print(k + 1, new_g)

                    if abs(prev_g / new_g - 1) < tol:
                        self.params = params
                        self.flat_hess_ = hess
                        return True

            elif method == "broyden":

                try:
                    grad = self.flat_grad(params)

                    if not (self.converged and
                            hasattr(self, 'flat_hess_') and
                            self.flat_hess_.shape[0] == grad.shape[0]):
                        self.flat_hess_ = self.flat_hess(params).data

                    J = pinvh(self.flat_hess_)
                    
                    c_len = len(self.constraint)
                    i1 = (-(c_len + 1) % grad.shape[0]) + 1

                    for k in range(max_steps):
                        prev_g = new_g
                        grad_prev = grad

                        step = -np.dot(J, grad)
                        flat_params += step[:i1]
                        params = self._reshape_params(flat_params)
                        
                        new_g = self.g(params)

                        if verbose > 0:
                            print(k + 1, new_g)

                        if abs(prev_g / new_g - 1) < tol:
                            self.params = params
                            self.flat_hess_ = pinvh(J)
                            self.converged = True
                            return True
                        
                        grad = self.flat_grad(params)
                        dgrad = grad - grad_prev
                        J += np.outer(step - np.dot(J, dgrad), dgrad) / \
                             (dgrad * dgrad).sum()

                except Exception as err:
                    self.converged = False
                    raise err
            
            else:
                raise ValueError("Invalid Method")

            self.converged = False
            raise ConvergenceError("Error: the objective function did not" +
                                   "converge after %d steps" % max_steps)