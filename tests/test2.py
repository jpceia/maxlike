import maxlike
import numpy as np
import pandas as pd
from sympy import symbols, lambdify, diff
from mpmath import findroot
from maxlike.func import Func, Sum, X, Linear
from maxlike.func_base import grad_tensor, hess_tensor
from maxlike.utils import prepare_dataframe, prepare_series

class SymbolicFunc(Func):
    def __init__(self, x, f_list):
        Df_list = [diff(f) for f in f_list]
        Hf_list = [diff(Df) for Df in Df_list]
        self.f = lambda u: np.stack(
            lambdify(x, f_list, 'numpy')(u)).swapaxes(0, -1)
        self.Df = lambda u: np.stack(
            lambdify(x, Df_list, 'numpy')(u)).swapaxes(0, -1)
        self.Hf = lambda u: np.stack(
            lambdify(x, Hf_list, 'numpy')(u)).swapaxes(0, -1)

    def __call__(self, params):
        return Tensor(self.f(params[0].values), dim=1)

    def grad(self, params, i):
        return grad_tensor(self.Df(params[0].values), params, i, True, dim=1)

    def hess(self, params, i, j):
        return hess_tensor(self.Hf(params[0].values), params, i, j, True, True, dim=1)


class Logistic(Func):
    def __call__(self, params):
        return Tensor(1 / (1 + np.exp(-params[0])))

    def grad(self, params, i):
        f = self.__call__(params).values
        return grad_tensor(f * (1 - f), params, i, True)

    def hess(self, params, i, j):
        f = self.__call__(params).values
        return hess_tensor(f * (1 - f) * (1 - 2 * f), params, i, j, True, True)

if __name__ == "__main__":
    x = symbols('x')
    g0 = x ** 4
    g1 = 4 * x ** 4 * (1-x)
    g2 = 10 * x ** 4 * (1-x) ** 2
    g3 = 20 * x ** 5 * (1-x) ** 3 / (1 - 2 * x * (1 - x))
    g = g0 + g1 + g2 + g3

    t0 = x ** 7
    t1 = 7 * x ** 7 * (1-x)
    t2 = 28 * x ** 7 * (1-x) ** 2
    t3 = 84 * x ** 7 * (1-x) ** 3
    t4 = 210 * x ** 7 * (1-x) ** 4
    t5 = 462 * x ** 7 * (1-x) ** 5
    t6 = 924 * x ** 8 * (1-x) ** 6 / (1-2*x*(1-x))
    t = t0 + t1 + t2 + t3 + t4 + t5 + t6

    s60 = g ** 6
    s61 = 6 * g ** 6 * (1-g)
    s62 = 21 * g ** 6 * (1-g) ** 2
    s63 = 56 * g ** 6 * (1-g) ** 3
    s64 = 126 * g ** 6 * (1-g) ** 4
    s75 = 252 * g ** 7 * (1-g) ** 5
    s76 = 504 * g ** 6 * (1-g) ** 6 * t
    s67 = 504 * g ** 6 * (1-g) ** 6 * (1-t)
    s57 = 252 * (1-g) ** 7 * g ** 5
    s46 = 126 * (1-g) ** 6 * g ** 4
    s36 = 56 * (1-g) ** 6 * g ** 3
    s26 = 21 * (1-g) ** 6 * g ** 2
    s16 = 6 * (1-g) ** 6 * g
    s06 = (1-g) ** 6
    s_list = [s06, s16, s26, s36, s46, s57, s67, s76, s75, s64, s63, s62, s61, s60]
    s_expectation = np.array([
        0.167222654998731,
        0.329429564250229,
        0.391283676196813,
        0.430437536668106,
        0.459073098050209,
        0.466058286268619,
        0.480953138435057,
        0.519046861564943,
        0.533941713731381,
        0.540926901949791,
        0.569562463331894,
        0.608716323803187,
        0.670570435749771,
        0.832777345001269])
    mass_s = sum([k * s for k, s in zip(s_expectation, s_list)])
    dd = lambdify(x, diff(mass_s))(.5)

    df = pd.read_csv("test_data_finite2.csv", sep=";", index_col=[0, 1]).stack()
    axis1 = list(set(df.index.levels[0]).union(set(df.index.levels[1])))
    axis = {'t1': axis1, 't2': axis1}
    N = prepare_series(df, add_axis=axis)[0]['N']
    R = N + np.flip(N.swapaxes(0, 1), 2)
    m = (R * s_expectation[None, None, :]).sum((1, 2)) / R.sum((1, 2))
    a = (0.5 - m) / dd

    proba = SymbolicFunc(x, s_list)
    s = Sum(2)
    s.add(X(), 0, 0)
    s.add(-X(), 0, 1)
    f = proba @ Logistic()
    mle = maxlike.Finite()
    mle.model = f @ s
    mle.add_constraint([0], Linear([1]))

    mle.add_param(a)
    tol = 1e-8

    mle.fit(tol=tol, verbose=True, N=N)
    a = mle.params_[0]
    s_a = mle.std_error()[0]
    print(pd.DataFrame({'a': a, 's_a': s_a}))