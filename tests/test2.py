import sys
sys.path.insert(0, "..")
import maxlike
import numpy as np
import pandas as pd
from sympy import symbols, lambdify, diff
from maxlike.func import Func, Sum, Encode, Linear
from maxlike.tensor import Tensor, grad_tensor, hess_tensor
from maxlike.utils import prepare_dataframe

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

    proba = SymbolicFunc(x, s_list)
    a = np.array([.1, .2, .3, .5, .6, .7, .8])
    s = Sum(2)
    s.add(Encode(), 0, 0)
    s.add(-Encode(), 0, 1)
    F = proba @ Logistic() @ s

    df = pd.read_csv("test_data_finite2.csv", sep=";", index_col=[0, 1]).stack().to_frame('g')
    df['w'] = 1
    L = 14
    n = max(df.index.levels[0].max(), df.index.levels[1].max())
    axis = {'t1': list(range(n)), 't2': list(range(n)), '': list(range(L))}
    N = prepare_dataframe(df, 'w', 'g', {'N': np.sum}, axis)[0]['N']

    mle = maxlike.Finite()
    mle.model = F
    mle.add_constraint([0], Linear([1]))

    idx1 = (N + N.swapaxes(0, 1)).sum(0)[:, 1:].sum(-1) != 0  # only 0:6
    idx2 = (N + N.swapaxes(0, 1)).sum(0)[:, :-1].sum(-1) != 0  # only 0:6
    idx = idx1 | idx2
    N = N[idx, :, :][:, idx, :]
    n = idx.sum()

    a = np.zeros(n)
    mle.add_param(a)
    tol = 1e-8

    mle.fit(tol=tol, verbose=True, N=N)
    a = mle.params_
    s_a = mle.std_error()
    print(pd.DataFrame({'a': a, 's_a': s_a}))
