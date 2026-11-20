import abc
import numpy as np
import scipy.sparse as sp
from functools import wraps
from collections import OrderedDict
from scipy.linalg import pinvh
from random import getrandbits
from .tensor import Tensor, SparseBlock
from .linalg import SymmetricSolver, SingularMatrix, DENSE_MAX_SIZE


def _dense_block(t, shape):
    """
    Values of a (gradient / hessian) block as an array with the given shape.
    A null derivative (Tensor(0), returned for parameters a function does not
    depend on) becomes an array of zeros.
    """
    values = np.asarray(getattr(t, "values", t))
    return np.broadcast_to(values, shape)


def _is_null(t):
    values = np.asarray(getattr(t, "values", t))
    return values.ndim == 0 and values == 0


def _converged(prev_g, g, tol):
    """Relative change of the objective below tol (safe for g == 0)."""
    diff = abs(g - prev_g)
    scale = abs(g)
    return diff < tol * scale if scale > 0 else diff < tol


class _SingularHessian(Exception):
    """The Newton system is (numerically) singular."""


class IdentifiabilityError(ValueError):
    """
    The objective is flat along some direction of the free parameters:
    they cannot all be estimated (a constraint or a fixed value is missing).
    """


class ConvergenceError(Exception):
    """
    The fit did not converge. `type` is "step" when no step increased the
    objective (no ascent direction, or damping failed), 0 when the maximum
    number of steps was reached.
    """
    def __init__(self, message, type=0):
        super(ConvergenceError, self).__init__(message)
        self.type = type


class Param(object):
    """
    An immutable parameter: an array of values with a boolean mask, True
    where the value is fixed (not estimated).

    It converts to its values with np.asarray, compressed() gives the free
    values, count() their number, and as_masked() a numpy masked array.
    Each instance hashes to a random id, so cached results computed for one
    set of parameters are not reused for another.
    """

    def __init__(self, values, mask=None):
        
        if mask is None or mask is False:
            mask = np.zeros_like(values, dtype=bool)
        else:
            assert mask.shape == values.shape

        object.__setattr__(self, 'values', np.asarray(values))
        object.__setattr__(self, 'mask',   np.asarray(mask))
        object.__setattr__(self, 'shape',  self.values.shape)
        object.__setattr__(self, 'hash',   getrandbits(128))
        self.values.flags["WRITEABLE"] = False

    def compressed(self):
        return self.values[~self.mask]

    def count(self):
        return (~self.mask).sum()

    def as_masked(self):
        return np.ma.array(self.values, mask=self.mask)

    def __array__(self, *args, **kwargs):
        return np.asarray(self.values)

    def __hash__(self):
        return self.hash

    def __setattr__(self, *ignored):
        raise NotImplementedError

    def __delattr__(self, *ignored):
        raise NotImplementedError


class MaxLike(abc.ABC):
    """
    Base class to estimate the maximum likelihood under different
    probabilistic models.
    """

    @abc.abstractmethod
    def __init__(self):
        # model related
        self.model = None

        self.constraint = []
        self.reg = []

        self.feat_dico = OrderedDict()
        self.N = None
        self.g_last = None
        self.converged = False

    @abc.abstractmethod
    def like(self, params, y):
        """
        Likelihood function.
        """

    @abc.abstractmethod
    def grad_like(self, params, y, der):
        """
        Calculates the gradient of the log-likelihood function.
        """

    @abc.abstractmethod
    def hess_like(self, params, y, der, hess):
        """
        Calculates the hessian of the log-likelihood function.
        """

    def _memo(self, name, key, compute):
        """
        Caches compute() while 'key' is the same object. Used for quantities
        that depend only on the model value y (e.g. exp(y), shared by like,
        grad_like and hess_like within a Newton step) or only on the data.
        'key' may be a tuple of objects, compared by identity.
        """
        if not isinstance(key, tuple):
            key = (key,)
        cached = self.__dict__.get("_memo_" + name)
        if cached is not None and len(cached[0]) == len(key) and \
                all(a is b for a, b in zip(cached[0], key)):
            return cached[1]
        value = compute()
        self.__dict__["_memo_" + name] = (key, value)
        return value

    def reset_params(self):
        self.params = ()
        self.param_names = []
        self.param_labels = []

    def g(self, params, y=None):
        """
        Objective function, to maximize.
        """
        if y is None:
            y = self.model(params)
        g = self.like(params, y)
        for param_map, h in self.reg:
            g -= h([params[k] for k in param_map])
        res = g / self._memo("n_total", self.N, self.N.sum)
        return float(res)

    def add_param(self, values, fixed=None, name=None, labels=None):
        """
        Adds a new parameter.

        Parameters
        ----------
        values: array-like, pd.Series or pd.DataFrame
            Initial guess for the parameter. The index (and columns) of a
            pandas object are used as labels of its entries.
        fixed: array-like (optional)
            Boolean array, True where the parameter has a fixed value.
        name: str (optional)
            Name of the parameter, used in summary(), get_params(labeled=True)
            and graph().
        labels: list (optional)
            Labels of the entries along each axis, e.g. [team_names].
        """
        from .inference import labels_from_values
        if labels is None:
            labels = labels_from_values(values)
        elif np.ndim(values) == 1 and all(np.isscalar(v) for v in labels):
            labels = [labels]   # labels of a single axis, as a flat list
        if fixed is not None and fixed is not False:
            fixed = np.asarray(fixed, dtype=bool)
        if not hasattr(self, "params"):
            self.reset_params()
        values = np.asarray(values, dtype=float)
        self.params = self.params + (Param(values, mask=fixed), )
        self.param_names.append(name)
        self.param_labels.append(labels)

    def get_params(self, labeled=False):
        """
        Current parameter values: masked arrays (fixed entries masked), or
        with labeled=True, pandas objects indexed by the parameter labels.
        """
        if labeled:
            from .inference import labeled as _labeled
            return [_labeled(self, k, p.as_masked())
                    for k, p in enumerate(self.params)]
        return [p.as_masked() for p in self.params]

    def summary(self, level=0.95):
        """
        Table of estimates with standard errors, z statistics, p-values and
        confidence intervals (normal approximation), one row per parameter
        entry, indexed by parameter name and label.
        """
        from .inference import summary
        return summary(self, level)

    def predict(self, model=None, params=None, se=False):
        """
        Value of the fitted model (or of another Func of the parameters,
        e.g. a win probability) at the fitted parameters. With se=True,
        returns (value, standard error) using the delta method.
        """
        from .inference import predict
        return predict(self, model, params, se)

    def simulate(self, rng=None, params=None):
        """
        Simulates a new data set from the fitted model, with the same
        counts N. Returns the keyword arguments of fit(), so a parametric
        bootstrap is other.fit(**mle.simulate(rng)).
        """
        from .inference import simulate
        return simulate(self, rng, params)

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

    def graph(self, param_names=None):
        """
        Computation graph of the problem: parameters, model, likelihood,
        constraints and regularizations (see maxlike.graph). Export it with
        .to_dot(), .to_mermaid() or .render(path).
        """
        from .graph import maxlike_graph
        return maxlike_graph(self, param_names)

    def akaike_information_criterion(self, nb_extra_params=0):
        """
        Akaike information criterion (AIC)

        Given a set of candidate models for the data,
        the preferred model is the one with the minimum AIC value.

        Simple version:
            AIC_0 = 2 * k - 2 ln L
                  = 2 * k - 2 * l * n

        Improved version:
            AIC = AIC_0 + 2 k * (k + 1) / (n - (k + 1))
                = 2 * n * (k / (n - k - 1) - l)

        This function returns:
            2 * k / (n - k - 1) - 2 * l


        k : degrees of freedom
        n : sample size
        l : mean log likelihood  (1/n) * log L

        """
        k = sum(p.count() for p in self.params)  # free params only
        k += nb_extra_params
        k -= len(self.constraint)
        n = self.N.sum(sum_val=True, sum_dim=True).values
        l = self.g(self.params)
        return 2 * k / (n - k - 1) - 2 * l

    # backward-compatible alias for the former (misspelled) name
    akaine_information_criterion = akaike_information_criterion

    def bayesian_information_criterion(self, nb_extra_params=0):
        """
        Bayesian information criterion (BIC)
        The model with the lowest BIC is preferred.

        BIC = ln(n) * k - 2 * ln L

        This function returns:
            k * ln(n) / n - 2 * l


        k : degrees of freedom
        n : sample size
        l : mean log likelihood  (1/n) * log L
        
        """
        k = sum(p.count() for p in self.params)  # free params only
        k += nb_extra_params
        k -= len(self.constraint)
        n = self.N.sum(sum_val=True, sum_dim=True).values
        l = self.g(self.params)
        return k * np.log(n) / n - 2 * l

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
            np.concatenate([p.values[p.mask] for p in self.params]))

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
        """
        Hessian of the objective, by blocks of parameters (dense arrays:
        (number of free parameters)^2 values, also when the Hessian was
        assembled as a sparse matrix).
        """
        hess = self.flat_hess_
        if sp.issparse(hess):
            hess = hess.toarray()
        return self._reshape_matrix(hess, 0)

    def error_matrix(self):
        """
        Covariance Matrix.

        It is dense, (number of free parameters)^2 values, whatever the
        sparsity of the Hessian: for large problems use std_error().
        """
        hess = self.flat_hess_
        if sp.issparse(hess):
            hess = hess.toarray()
        try:
            mask = hess.mask
        except AttributeError:
            mask = None
        return self._reshape_matrix(
            -np.ma.array(pinvh(hess.data), mask=mask))

    def _sparse_solver(self, hess):
        """
        Factorisation of a sparse (bordered) Hessian. The entries of each
        parameter are candidates for elimination (see maxlike.linalg).
        """
        stops = np.cumsum([int(p.count()) for p in self.params])
        return SymmetricSolver(
            hess, list(zip(np.concatenate([[0], stops[:-1]]), stops)))

    def _inverse_diagonal(self, hess):
        """Diagonal of the inverse of the (bordered) Hessian"""
        if sp.issparse(hess):
            return self._sparse_solver(hess).inverse_diagonal()
        return np.diag(np.linalg.inv(hess))

    def std_error(self):
        """
        Standard Deviation Array.

        With a sparse Hessian (see flat_hess) only the diagonal of the
        inverse is computed, from the factorisation used by the Newton
        steps (see maxlike.linalg for its cost).
        """
        var = -self._inverse_diagonal(self.flat_hess_)
        # drop the entries of the Lagrange multipliers (constraints)
        var = var[:var.size - len(self.constraint)]
        return [p.as_masked() for p in self._reshape_array(np.sqrt(var))]

    def flat_grad(self, params, *args, **kwargs):

        n = len(params)
        grad = self.grad_like(params, *args, **kwargs)
        grad += [0] * len(self.constraint)

        for k, (param_map, gamma, g) in enumerate(self.constraint):
            args = [params[k] for k in param_map]
            grad_g = g.grad(args)
            for i, idx in enumerate(param_map):
                grad[idx] += gamma * grad_g[i]
                grad[n + k] = [g(args)]

        for param_map, h in self.reg:
            args = [params[k] for k in param_map]
            grad_h = h.grad(args)
            for i, idx in enumerate(param_map):
                grad[idx] -= grad_h[i]

        grad = [_dense_block(d, p.shape)[~p.mask]
                for d, p in zip(grad, params)] + grad[n:]
        grad = np.concatenate(grad)

        return grad

    # Assembly of the Hessian as a scipy.sparse matrix, solved by a sparse
    # factorisation. None: when the model gives sparse blocks (observation
    # indexed models, see maxlike.func.Take) and there are at least
    # _SPARSE_MIN_SIZE free parameters; True / False: always / never.
    sparse_hess = None
    _SPARSE_MIN_SIZE = 500

    def _use_sparse_hess(self, params, hess):
        if self.sparse_hess is not None:
            return bool(self.sparse_hess)
        n_free = sum(int(p.count()) for p in params)
        blocks = [h for h_line in hess for h in h_line
                  if isinstance(h, SparseBlock)]
        if not blocks or n_free < self._SPARSE_MIN_SIZE:
            return False
        # blocks that are (nearly) full, e.g. when every pair of teams met,
        # are cheaper to assemble and solve as a dense matrix
        stored = sum(h.matrix.nnz for h in blocks)
        return stored < .25 * n_free ** 2 or n_free > DENSE_MAX_SIZE

    def _flat_hess_sparse(self, params, hess):
        """
        The same matrix as flat_hess, as a scipy.sparse (CSC) matrix, from
        the blocks returned by hess_like. Sparse blocks are never expanded.
        """
        c_len = len(self.constraint)
        sizes = [int(np.prod(p.shape, dtype=int)) for p in params]
        offset = np.concatenate([[0], np.cumsum([p.count() for p in params])])
        n_free = int(offset[-1])

        # position of every entry among the free ones (-1 if fixed)
        free_pos = []
        for p in params:
            mask = np.asarray(p.mask).ravel()
            if mask.any():
                pos = np.cumsum(~mask) - 1
                pos[mask] = -1
                free_pos.append(pos)
            else:
                free_pos.append(None)

        rows, cols, data = [], [], []

        def add(i, j, block, sign=1):
            # block: derivative with respect to parameters i and j, or with
            # respect to the multiplier of constraint i - len(params) and
            # parameter j; the transposed block is added too
            if isinstance(block, SparseBlock):
                block = block.matrix
            if block is None or \
                    (not sp.issparse(block) and _is_null(block)):
                return
            if i < len(params) and i < j:
                i, j = j, i
                block = block.transpose()
            if not sp.issparse(block):
                n_rows = sizes[i] if i < len(params) else 1
                block = np.broadcast_to(
                    np.asarray(getattr(block, "values", block)),
                    (params[i].shape if i < len(params) else ()) +
                    params[j].shape).reshape((n_rows, sizes[j]))
            block = sp.coo_matrix(block)
            r, c, d = block.row, block.col, sign * block.data
            if i < len(params):
                if free_pos[i] is not None:
                    r = free_pos[i][r]
                r = r + offset[i]
            else:
                r = r + (n_free + i - len(params))
            if free_pos[j] is not None:
                c = free_pos[j][c]
                keep = c >= 0
                if i < len(params) and free_pos[i] is not None:
                    keep &= r >= offset[i]
                r, c, d = r[keep], c[keep], d[keep]
            elif i < len(params) and free_pos[i] is not None:
                keep = r >= offset[i]
                r, c, d = r[keep], c[keep], d[keep]
            c = c + offset[j]
            rows.append(r)
            cols.append(c)
            data.append(d)
            if i != j:
                rows.append(c)
                cols.append(r)
                data.append(d)

        def add_func(foo, param_map, sign):
            # second derivatives of a constraint / regularization
            args = [params[k] for k in param_map]
            sparse = getattr(foo, "hess_sparse", None)
            for i, idx in enumerate(param_map):
                for j in range(i + 1):
                    add(idx, param_map[j], sparse(args, i, j) if sparse
                        else foo.hess(args, i, j), sign)

        for i in range(len(params)):
            for j in range(i + 1):
                add(i, j, hess[i][j])

        for k, (param_map, gamma, g) in enumerate(self.constraint):
            args = [params[k] for k in param_map]
            grad_g = g.grad(args)
            for i, idx in enumerate(param_map):
                add(len(params) + k, idx, grad_g[i].sum())
            if gamma != 0:
                add_func(g, param_map, gamma)

        for param_map, h in self.reg:
            add_func(h, param_map, -1)

        if not data:
            return sp.csc_matrix((n_free + c_len, n_free + c_len))
        return sp.csc_matrix(
            (np.concatenate(data),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_free + c_len, n_free + c_len))

    def flat_hess(self, params, *args, **kwargs):
        """
        Hessian of the objective with respect to the free parameters,
        bordered by the gradients of the constraints (KKT matrix).

        A dense array, or a scipy.sparse matrix when the model gives sparse
        blocks and the problem is large enough (see sparse_hess).
        """
        n = len(params)
        c_len = len(self.constraint)

        # --------------------------------------------------------------------
        # 1st phase: Evaluate and sum
        # --------------------------------------------------------------------
        hess = self.hess_like(params, *args, **kwargs)

        if self._use_sparse_hess(params, hess):
            return self._flat_hess_sparse(params, hess)

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
                hess_c[k][idx] += grad_g[i].sum().values
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

        hess = [[_dense_block(hess[i][j], p_i.shape + params[j].shape)[
                     np.multiply.outer(~p_i.mask, ~params[j].mask)].
                     reshape((p_i.count(), params[j].count()))
                 for j in range(i + 1)]
                for i, p_i in enumerate(params)]
        
        hess = [[hess[i][j] for j in range(i)] +
                [hess[j][i].transpose() for j in range(i, n)]
                for i in range(n)]
        
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
     
    def _post_fit_checks(self, params):
        """Hook for likelihood-specific checks after a fit (warnings)."""

    def _check_params_used(self, params, der):
        """
        Raises a ValueError if a parameter with free values does not affect
        the objective: the model does not depend on it and no regularization
        applies to it, so it cannot be estimated.
        """
        regularized = {k for param_map, _ in self.reg for k in param_map}
        unused = [i for i, (p, d) in enumerate(zip(params, der))
                  if p.count() > 0 and _is_null(d) and i not in regularized]
        if unused:
            raise ValueError(
                "parameter(s) %s are not used by the model nor by any "
                "regularization, so they cannot be estimated; fix all their "
                "values or remove them" % ", ".join(map(str, unused)))

    # relative residual above which a solution of the Newton system is
    # considered unreliable (numerically singular system)
    _SINGULAR_RESIDUAL = 1e-6

    def _solve_newton(self, lhess, lgrad):
        """
        Solves the (bordered) Newton system. Raises _SingularHessian when it
        is singular: exactly (LAPACK reports it) or numerically (non-finite
        solution or large residual, a cheap matrix-vector check).
        """
        if sp.issparse(lhess):
            # sparse direct solver (see maxlike.linalg)
            try:
                step = self._sparse_solver(lhess).solve(lgrad)
            except SingularMatrix:
                raise _SingularHessian
        else:
            try:
                step = np.linalg.solve(lhess, lgrad)
            except np.linalg.LinAlgError:
                raise _SingularHessian
        if not np.all(np.isfinite(step)):
            raise _SingularHessian
        residual = np.abs(lhess.dot(step) - lgrad).max()
        if residual > self._SINGULAR_RESIDUAL * max(np.abs(lgrad).max(), 1e-300):
            raise _SingularHessian
        return step

    # reciprocal condition number below which the converged (bordered)
    # Hessian is treated as singular
    _SINGULAR_RCOND = 1e-12

    def _check_identified(self, params, lhess):
        """
        Once at convergence: if the bordered Hessian is singular, the
        parameters are not identifiable (the objective is flat along some
        direction). Costs one Newton solve per fit (disable with
        fit(check_identifiability=False)).
        """
        if not getattr(self, "_check_identifiability", True):
            return
        if lhess.shape[0] == 0:
            return
        # one solve for a random right-hand side: near a singular matrix the
        # solution blows up along the flat direction, and |K| |x| / |b|
        # estimates 1 / rcond. Uses the same solver as the Newton steps (the
        # sparse factorisation, or numpy's LAPACK: calling scipy's separate
        # BLAS here would leave its threads competing with numpy's)
        anorm = abs(lhess).sum(0).max()
        b = np.random.default_rng(0).standard_normal(lhess.shape[0])
        try:
            if sp.issparse(lhess):
                x = self._sparse_solver(lhess).solve(b)
            else:
                x = np.linalg.solve(np.asarray(lhess, dtype=float), b)
            rcond = np.abs(b).max() / (anorm * np.abs(x).max())
        except (SingularMatrix, np.linalg.LinAlgError):
            rcond = 0.
        if np.isfinite(rcond) and rcond > self._SINGULAR_RCOND:
            return
        directions = self._null_directions(params, lhess)
        if directions:
            self.converged = False
            raise IdentifiabilityError(
                "the parameters are not identifiable: the objective is "
                "flat along %s. Add a constraint (add_constraint) or fix "
                "some values." % "; and along ".join(directions))

    def _newton_direction(self, params, lgrad, lhess):
        """
        Newton step for the free parameters (new = old - step). If the step
        does not increase the objective (Hessian not negative definite),
        it is damped Levenberg-Marquardt style: lambda * I is subtracted
        from the parameter block until it does. A singular system caused by
        non-identifiable parameters raises IdentifiabilityError.
        """
        n_free = lgrad.size - len(self.constraint)
        grad = lgrad[:n_free]

        def is_ascent(step):
            d = -step[:n_free]
            return grad.dot(d) >= -1e-10 * np.linalg.norm(grad) * \
                np.linalg.norm(d)

        flat = np.concatenate([p.compressed() for p in params]) \
            if n_free else np.zeros(0)
        scale = 1 + (np.abs(flat).max() if flat.size else 0.)
        try:
            step = self._solve_newton(lhess, lgrad)
            # a numerically singular system can still be solved with a small
            # residual, giving an absurdly large step along a flat direction
            if np.abs(step[:n_free]).max(initial=0) > 1e6 * scale:
                raise _SingularHessian
            if is_ascent(step):
                return step[:n_free]
        except _SingularHessian:
            directions = self._null_directions(params, lhess)
            if directions:
                raise IdentifiabilityError(
                    "the parameters are not identifiable: the objective is "
                    "flat along %s. Add a constraint (add_constraint) or fix "
                    "some values." % "; and along ".join(directions))

        # make the parameter block negative definite: shift it by about
        # twice its largest eigenvalue (exact for moderate sizes, else the
        # Gershgorin bound), then increase the shift if still needed
        if sp.issparse(lhess):
            H = lhess.tocsr()[:n_free, :n_free]
            diag = H.diagonal()
            abs_sum = np.asarray(abs(H).sum(1)).ravel()
        else:
            H = np.asarray(lhess)[:n_free, :n_free]
            diag = np.diag(H)
            abs_sum = None
        scale = np.abs(diag).max() if n_free else 1.
        scale = scale if scale > 0 else 1.
        if n_free <= 2000:
            top = np.linalg.eigvalsh(
                H.toarray() if sp.issparse(H) else H)[-1]
        else:
            if abs_sum is None:
                abs_sum = np.abs(H).sum(1)
            top = (diag + abs_sum - np.abs(diag)).max()
        lam = 2 * max(top, 0) + 1e-8 * scale
        shift = np.zeros(lgrad.size)
        shift[:n_free] = 1
        for _ in range(60):
            try:
                damping = sp.diags(shift) if sp.issparse(lhess) \
                    else np.diag(shift)
                step = self._solve_newton(lhess - lam * damping, lgrad)
                if is_ascent(step):
                    return step[:n_free]
            except _SingularHessian:
                pass
            lam *= 2
        raise ConvergenceError("could not find an ascent direction", "step")

    def _null_directions(self, params, lhess, rtol=1e-8, max_params=5000):
        """
        Directions of the free parameters (compatible with the constraints)
        along which the Hessian vanishes, described in words.
        """
        from scipy.linalg import null_space, eigh
        from .inference import axis_labels, param_name
        n_free = lhess.shape[0] - len(self.constraint)
        if n_free > max_params:
            return []
        if sp.issparse(lhess):
            lhess = lhess.toarray()
        lhess = np.asarray(lhess, dtype=float)
        H = lhess[:n_free, :n_free]
        C = lhess[n_free:, :n_free]
        Z = null_space(C) if C.shape[0] else np.eye(n_free)
        if Z.shape[1] == 0:
            return []
        w, v = eigh(Z.T @ H @ Z)
        scale = np.abs(w).max()
        null = np.abs(w) <= rtol * scale if scale > 0 else np.ones_like(w,
                                                                        bool)
        # (parameter, label) of each free entry, in optimiser order
        entries = []
        for k, p in enumerate(params):
            axes = axis_labels(self, k)
            for idx in zip(*np.nonzero(~np.asarray(p.mask))) if p.shape \
                    else ([()] if p.count() else []):
                label = ", ".join(str(ax[i]) for ax, i in zip(axes, idx))
                entries.append((k, label))
        descriptions = []
        for u in v[:, null].T:
            d = Z @ u
            d = d / d[np.argmax(np.abs(d))]
            parts = []
            for k in range(len(params)):
                comp = np.array([d[i] for i, (kk, _) in enumerate(entries)
                                 if kk == k])
                if comp.size == 0 or np.abs(comp).max() < 1e-6:
                    continue
                name = param_name(self, k)
                if comp.size > 1 and np.ptp(comp) < 1e-6:
                    parts.append("all of '%s' (%+.3g)" % (name, comp[0]))
                    continue
                labels = [lab for kk, lab in entries if kk == k]
                top = np.argsort(-np.abs(comp))[:3]
                parts.append(", ".join(
                    "'%s'%s (%+.3g)" % (name, "[%s]" % labels[i]
                                        if labels[i] else "", comp[i])
                    for i in top if abs(comp[i]) >= 1e-6))
            descriptions.append(" + ".join(parts))
        return descriptions

    def identifiability(self, params=None):
        """
        Checks that the free parameters can be estimated: returns the
        directions (described in words) along which the objective is flat
        at the given (default: current) parameters, i.e. an empty list when
        the model is identifiable there.
        """
        params = self.params if params is None else params
        y, der, hess = self.model.eval(params)
        lhess = self.flat_hess(params, y, der, hess)
        return self._null_directions(params, lhess)

    def newton_step(self, params):
        
        y, der, hess = self.model.eval(params)
        self._check_params_used(params, der)

        g = self.g(params, y)
        lgrad = self.flat_grad(params, y, der)
        lhess = self.flat_hess(params, y, der, hess)
        
        step = self._newton_direction(params, lgrad, lhess)
        return step, g, lgrad, lhess

    def fit(self, tol=1e-8, max_steps=20, verbose=0, method="Newton",
            check_identifiability=True, **kwargs):
        """
        Run the algorithm to find the best fit.

        Parameters
        ----------
        tol : float (optional)
            Tolerance for termination.

        max_steps : int (optional)
            Maximum number of steps that the algorithm will perform.

        verbose : int (optional)
            Sets the verbosity of the algorithm.
            0 : no messages (default)
            1 : low verbosity
            2 : high verbosity

        method: string (optional)
            Method to calibrate the model
            * newton
            * broyden

        check_identifiability: bool (optional)
            At convergence, check that the (bordered) Hessian is not
            singular and raise IdentifiabilityError naming the flat
            directions otherwise. Costs about one extra Newton solve.

        Returns
        -------
        out : boolean
            Returns True if the algorithm converges, otherwise returns
            False.
        """

        self._check_identifiability = check_identifiability
        dim = getattr(self, 'dim', 0)
        for k in kwargs:
            self.__dict__[k] = Tensor(kwargs[k], dim=dim)

        method = method.lower()

        # --------------------------------------------------------------------
        # Flattening Params
        # --------------------------------------------------------------------

        params = self.params
        flat_params = np.concatenate([p.compressed() for p in params])
    
        if method == "newton":

            # halvings of the Newton step in the line search; only used when
            # the full step does not increase the objective
            max_inner_steps = 30
            for k in range(max_steps):
                step, g, grad, hess = self.newton_step(params)
                prev_g = g

                if verbose > 0:
                    print(k, g)

                mult = 1
                for _ in range(max_inner_steps):
                    new_flat_params = flat_params - mult * step
                    params = self._reshape_params(new_flat_params)
                    g = self.g(params, self.model(params))
                    if g >= prev_g:
                        flat_params = new_flat_params
                        break
                    else:
                        mult *= .5
                        if verbose > 1:
                            print("\tmultiplier = ", mult)
                else:
                    # no increase: at the optimum the step is negligible
                    # (rounding prevents any increase); otherwise give up
                    size = np.abs(step).max() if step.size else 0.
                    if size <= 1e-8 * (1 + np.abs(flat_params).max()):
                        params = self._reshape_params(flat_params)
                        self._check_identified(params, hess)
                        self.params = params
                        self.flat_hess_ = hess
                        self.converged = True
                        self._post_fit_checks(params)
                        return True
                    self.converged = False
                    raise ConvergenceError(
                        "Error: the objective function did not increase " +
                        "after %d steps" % max_inner_steps, "step")

                if _converged(prev_g, g, tol):
                    self._check_identified(params, hess)
                    self.params = params
                    self.flat_hess_ = hess
                    self.converged = True
                    self._post_fit_checks(params)
                    return True

        elif method == "broyden":

            g = self.g(params)

            if verbose > 0:
                print(0, g)

            try:
                y, der, hess = self.model.eval(params)
                self._check_params_used(params, der)
                grad = self.flat_grad(params, y, der)

                if not (self.converged and
                        hasattr(self, 'flat_hess_') and
                        self.flat_hess_.shape[0] == grad.shape[0]):
                    h = self.flat_hess(params, y, der, hess)
                    # Broyden updates a dense inverse: densify a sparse
                    # Hessian
                    self.flat_hess_ = h.toarray() if sp.issparse(h) \
                        else np.asarray(h)

                J = pinvh(self.flat_hess_)
                
                c_len = len(self.constraint)
                i1 = (-(c_len + 1) % grad.shape[0]) + 1

                for k in range(max_steps):
                    prev_g = g
                    grad_prev = grad

                    step = -np.dot(J, grad)
                    flat_params += step[:i1]
                    params = self._reshape_params(flat_params)
                    g = self.g(params)

                    if verbose > 0:
                        print(k + 1, g)

                    if _converged(prev_g, g, tol):
                        self.params = params
                        self.flat_hess_ = pinvh(J)
                        self.converged = True
                        return True
                    
                    y, der = self.model(params), self.model.grad(params)
                    grad = self.flat_grad(params, y, der)
                    dgrad = grad - grad_prev
                    J += np.outer(step - np.dot(J, dgrad), dgrad) / \
                         (dgrad * dgrad).sum()

            except Exception as err:
                self.converged = False
                raise err
        
        else:
            raise ValueError("Invalid Method")

        self.converged = False
        self._post_fit_checks(params)
        raise ConvergenceError("Error: the objective function did not " +
                               "converge after %d steps" % max_steps)
