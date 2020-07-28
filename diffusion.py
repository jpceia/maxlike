from maxlike.func.func_base import Func, grad_tensor, hess_tensor, null_func
from maxlike.tensor import Tensor
import maxlike
from maxlike.func import (
    Vector, Linear, Quadratic, Exp, Log, Constant, Scalar, 
    Poisson, NegativeBinomial, Sum, Product)
from maxlike.func import X as XX
from array import array
import numpy as np


class Diffusion(Func):
    """
    Function to model product diffusion / canibalization
    The model is given by
    y_i = q_i + q_o @ (1 - T_oo)^-1 @ T_oi

    q : array of default quantity sold (if all the products are available)
    T : sales transference.
        If the product i is not available, the sales of the product j
        are going to increase q_i * T_ij
    o : corresponds to the set of out of stock products
    """

    def __init__(self, avail_matrix):
        assert avail_matrix.ndim == 2 # shape = N_o * N_q
        self.idx = avail_matrix.T

    def __call__(self, params): # confirmed
        q, T = [np.asarray(p) for p in params]
        assert q.ndim == 1
        assert T.ndim == 2
        assert self.idx.shape[0] == q.size

        y = np.zeros_like(self.idx, dtype=np.float)
        for k, a in enumerate(self.idx.T):
            oo = np.ix_(~a, ~a)
            oa = np.ix_(~a,  a)
            n_o = (~a).sum()
            q_o = q[~a]
            I_oo = np.eye(n_o)
            y[a, k] = q[a] + q_o @  np.linalg.solve(I_oo - T[oo], T[oa])
        return Tensor(y, dim=1)

    def grad(self, params, i):
        q, T = [np.asarray(p) for p in params]
        assert q.ndim == 1
        assert T.ndim == 2
        assert self.idx.shape[0] == q.size

        if i == 0: # confirmed
            
            D_q = np.zeros(q.shape + self.idx.shape)

            for k, a in enumerate(self.idx.T):
                aa = np.ix_(a, a)
                oa = np.ix_(~a, a)
                oo = np.ix_(~a, ~a)
                I_aa = np.eye(a.sum())
                I_oo = np.eye((~a).sum())
                D_q[:, :, k][aa] = I_aa
                D_q[:, :, k][oa] = np.linalg.solve(I_oo - T[oo], T[oa])
            return grad_tensor(D_q, params, 0, dim=1)

        if i == 1: # confirmed
            
            D_T = np.zeros(T.shape + self.idx.shape)
            
            for k, a in enumerate(self.idx.T):
                oa  = np.ix_(~a,  a)
                oo  = np.ix_(~a, ~a)
                ooa = np.ix_(~a, ~a, a)
                oaa = np.ix_(~a,  a, a)

                I_oo = np.eye((~a).sum())

                O = np.linalg.inv(I_oo - T[oo])
                q_o = q[~a]
                I_aa = np.eye(a.sum())

                D_T[:, :, :, k][ooa] = (q_o @ O)[:, None, None] * (O @ T[oa])
                D_T[:, :, :, k][oaa] = (q_o @ O)[:, None, None] * I_aa
            return grad_tensor(D_T, params, 1, dim=1)

        raise ValueError("i must be 0 or 1")

    def hess(self, params, i, j):
        q, T = [np.asarray(p) for p in params]
        assert q.ndim == 1
        assert T.ndim == 2
        assert self.idx.shape[0] == q.size

        if i == 0 and j == 0:
            return Tensor(0)

        if i == 0 and j == 1: # confirmed
            H_qT = np.zeros(q.shape + T.shape + self.idx.shape)
            
            for k, a in enumerate(self.idx.T):
                oa = np.ix_(~a,  a)
                oo = np.ix_(~a, ~a)
                oooa = np.ix_(~a, ~a, ~a, a)
                ooaa = np.ix_(~a, ~a,  a, a)

                I_oo = np.eye((~a).sum())
                O = np.linalg.inv(I_oo - T[oo])
                I_aa = np.eye(a.sum())

                H_qT[:, :, :, :, k][oooa] = O[:, :, None, None] * (O @ T[oa])
                H_qT[:, :, :, :, k][ooaa] = O[:, :, None, None] * I_aa
            return hess_tensor(H_qT, params, 0, 1, dim=1)

        if i == 1 and j == 1:
            H_TT = np.zeros(T.shape * 2 + self.idx.shape)

            for k, a in enumerate(self.idx.T):
                oa = np.ix_(~a,  a)
                oo = np.ix_(~a, ~a)
                ooooa = np.ix_(~a, ~a, ~a, ~a, a)
                oooaa = np.ix_(~a, ~a, ~a,  a, a)
                oaooa = np.ix_(~a,  a, ~a, ~a, a)

                I_oo = np.eye((~a).sum())
                O = np.linalg.inv(I_oo - T[oo])
                I_aa = np.eye(a.sum())
                q_o = q[~a]
                H_TT[:, :, :, :, :, k][ooooa] = \
                    (q_o @ O)[:, None, None, None, None] * O[:, :, None, None] * (O @ T[oa]) + \
                    (q_o @ O)[:, None, None] * O.T[:, None, None, :, None] * (O @ T[oa])[:, None, None, :]

                H_TT[:, :, :, :, :, k][oooaa] = \
                    (q_o @ O)[:, None, None, None, None] * O[:, :, None, None] * I_aa

                H_TT[:, :, :, :, :, k][oaooa] = H_TT[:, :, :, :, :, k][oooaa].swapaxes(0, 2).swapaxes(1, 3)

            return hess_tensor(H_TT, params, 1, 1, dim=1)

    def eval(self, params):
        q, T = [np.asarray(p) for p in params]
        assert q.ndim == 1
        assert T.ndim == 2
        assert self.idx.shape[0] == q.size

        y = np.zeros_like(self.idx, dtype=np.float)
        D_q = np.zeros(q.shape + self.idx.shape)
        D_T = np.zeros(T.shape + self.idx.shape)
        H_qT = np.zeros(q.shape + T.shape + self.idx.shape)
        H_TT = np.zeros(T.shape * 2 + self.idx.shape)

        for k, a in enumerate(self.idx.T):
            
            aa = np.ix_(a, a)
            oa = np.ix_(~a, a)
            oo = np.ix_(~a, ~a)

            I_aa = np.eye(a.sum())
            I_oo = np.eye((~a).sum())
            O = np.linalg.inv(I_oo - T[oo])
            q_o = q[~a]

            # call
            y[a, k] = q[a] + q_o @ O @ T[oa]

            # D_q
            D_q[:, :, k][aa] = I_aa
            D_q[:, :, k][oa] = O @ T[oa]

            # D_T
            ooa = np.ix_(~a, ~a, a)
            oaa = np.ix_(~a,  a, a)
            D_T[:, :, :, k][ooa] = (q_o @ O)[:, None, None] * (O @ T[oa])
            D_T[:, :, :, k][oaa] = (q_o @ O)[:, None, None] * I_aa

            # H_qT
            oooa = np.ix_(~a, ~a, ~a, a)
            ooaa = np.ix_(~a, ~a,  a, a)
            H_qT[:, :, :, :, k][oooa] = O[:, :, None, None] * (O @ T[oa])
            H_qT[:, :, :, :, k][ooaa] = O[:, :, None, None] * I_aa

            # H_TT
            ooooa = np.ix_(~a, ~a, ~a, ~a, a)
            oooaa = np.ix_(~a, ~a, ~a,  a, a)
            oaooa = np.ix_(~a,  a, ~a, ~a, a)
            H_TT[:, :, :, :, :, k][ooooa] = \
                (q_o @ O)[:, None, None, None, None] * O[:, :, None, None] * (O @ T[oa]) + \
                (q_o @ O)[:, None, None] * O.T[:, None, None, :, None] * (O @ T[oa])[:, None, None, :]

            H_TT[:, :, :, :, :, k][oooaa] = \
                (q_o @ O)[:, None, None, None, None] * O[:, :, None, None] * I_aa

            H_TT[:, :, :, :, :, k][oaooa] = H_TT[:, :, :, :, :, k][oooaa].swapaxes(0, 2).swapaxes(1, 3)

        return Tensor(y, dim=1), \
            [grad_tensor(D_q, params, 0, dim=1), grad_tensor(D_T, params, 1, dim=1)], \
            [[Tensor(0)], [hess_tensor(H_qT, params, 0, 1, dim=1).transpose(),
              hess_tensor(H_TT, params, 1, 1, dim=1)]]


if __name__ == "__main__":


    """
    T = np.array([
            [0,   0.2, 0.1],
            [0.1, 0,   0.3],
            [0.5, 0.6,   0]
        ], dtype=np.float)

    avail_matrix = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.bool)

    q = np.array([6, 7, 8], dtype=np.float)
    d = Diffusion(avail_matrix)
    y = d.eval([q, T])[2][1][0]

    print(y)
    print(y.values.shape)


    k = ([2],)
    ij = ([0], [1])
    kj = ([2], [0])

    de = 0.0001
    q = np.array([6, 7, 8], dtype=np.float)
    dq = np.zeros(3)
    dq[k] = de

    dT = np.zeros_like(T)
    dT2 = np.zeros_like(T)


    dT[ij] = de
    dT2[kj] = de
    d = Diffusion(avail_matrix)
    y = d([q, T])

    dy = (d([q, T + dT]) - y) / de
    ddy = (d([q, T+dT+dT2]) - d([q, T+dT2]) - d([q, T + dT]) + y) / de / de
    #print(ddy)

    #print(d.hess([q, T], 1, 1)[ij + kj])

    print(d.eval([q, T]))


    Model
    mle.model = Sum(3)
    mle.add(Log() @ Diffusion(idx), [0, 1], [0, 1])
    mle.add(X(), 2, 2)
    mle.add(X(), 3, 3)

    mle.contraint(Linear() @ Exp(), 2)
    mle.constraint(Linear(), 3)

    q = np.ones(5)
    T = np.zeros((5, 5))
    T_fix = np.eye(5)
    S = np.zeros(300)
    Q = np.ones(100)
    mle.add_param(q)
    mle.add_param(T, T_fix)
    mle.add_param(S)
    mle.add_param()

    """
    import pandas as pd

    df_sls = pd.read_csv("sales2020.csv", header=[0, 1], index_col=[0])
    df_stk = pd.read_csv("stock2020.csv", header=[0, 1], index_col=[0])
    n_T = len(df_sls.index)
    n_N = len(df_sls.columns.get_level_values(0).unique())
    n_M = len(df_sls.columns.get_level_values(1).unique())
    X = df_sls.values.reshape((n_T, n_N, n_M))
    S = df_stk.values.reshape((n_T, n_N, n_M))
    A = (X > 0) | (S == 0)
    O = np.unique(A.swapaxes(1, 2).reshape((n_T * n_M, n_N)), axis=0)
    N = O[:, None, :, None] == A[None, :, :, :]
    X = N * X
    N = N.sum(1)
    X = X.sum(1)

    mle = maxlike.Poisson()
    mle.model = Sum(3)
    mle.model.add(Log() @ Diffusion(O), [0, 1], [0, 1])
    mle.model.add(XX(), 2, 2)

    mle.add_constraint([2], Linear(1) @ Exp())

    q_guess = np.log(X.sum((0, 2)) / N.sum((0, 2)))
    Q_guess = X.sum((0, 1)) / N.sum((0, 1))
    Q_guess /= Q_guess.sum()
    Q_guess = np.log(Q_guess)
    T_guess = np.zeros((n_N, n_N))
    T_fix = np.eye(n_N, dtype=np.bool)
    T_ineq = np.ones_like(T_guess, dtype=np.bool)
    Q = np.ones(n_M)
    mle.add_param(q_guess)
    mle.add_param(T_guess, T_fix, T_ineq)
    mle.add_param(Q_guess)

    mle.fit(X=X, N=N)
    """

    ## checks that sum from DATES is correct
    #print(X.sum((1, 2)))
    #print(df_sls.sum(1))

    ## checks that sum from SKU is correct
    #print(X.sum((0, 2)))
    #print(df_sls.sum().sum(level=0))

    ## checks that sum from STORE is correct
    #print(X.sum((0, 1)))
    #print(df_sls.sum().sum(level=1))