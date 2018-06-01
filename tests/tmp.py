import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "..")
import maxlike
from maxlike.func import X, Linear, Constant, Scalar, Sum, Product


if __name__ == '__main__':
    mle = maxlike.Poisson()
    s = Sum(2).add(X(), 0, 0).add(-X(), 1, 1)
    s_diff = Sum(2)
    s_diff.add(s, [0, 1], [0, 1])
    s_diff.add(-s, [0, 1], [1, 0])
    hs = Product(2)
    hs.add(Scalar(), 2, None)
    hs.add(s_diff, [0, 1], [0, 1])
    h_diff = Sum(2)
    h_diff.add(Scalar(), 2, None)
    h_diff.add(hs, [0, 1, 3], [0, 1])
    H = Product(3)
    H.add(Constant(np.arange(2) - .5), [], 2)
    H.add(h_diff, [0, 1, 2, 3], [0, 1])
    F = Sum(3)
    F.add(s, [0, 1], [0, 1])
    F.add(H, [0, 1, 2, 3], [0, 1, 2])
    mle.model = F
    mle.add_constraint([0, 1], Linear([1, 1]))
    g = pd.read_csv("test_data1.csv", index_col=[0, 1, 2])['g']
    prepared_data, _ = maxlike.utils.prepare_series(
        g, {'N': np.size, 'X': np.sum})
    _h = g.groupby(level='h').mean().map(np.log).reset_index().prod(1).sum()
    _h1 = .1
    log_mean = np.log(g.mean()) / 2
    _a = g.groupby(level='t1').mean().map(np.log) - log_mean
    _b = log_mean - g.groupby(level='t2').mean()
    mle.add_param(_a.values)
    mle.add_param(_b.values)
    mle.add_param(_h, False)
    mle.add_param(_h1, False)
    tol = 1e-8
    mle.fit(tol=tol, scipy=True, verbose=True, **prepared_data)
    mle.reset_params()
    mle.add_param(_a.values)
    mle.add_param(_b.values)
    mle.add_param(_h, False)
    mle.add_param(_h1, False)

    mle.params_[0][:] = np.zeros(20)
    mle.params_[1][:] = np.zeros(20)
    mle.N.values[:,:,:] = 1

    flat_params = np.concatenate([p[~p.mask] for p in mle.params_])
    flat_der = np.concatenate([d.values[~p.mask] for d, p in zip(mle.grad_like(mle.params_), mle.params_)])
    hess = mle.hess_like(mle.params_)
    flat_hess = [[hess[j][i].values[np.multiply.outer(~mle.params_[i].mask, ~p_j.mask)].reshape((mle.params_[i].count(), p_j.count())) for i in range(j + 1)] for j, p_j in enumerate(mle.params_)]
    flat_hess = [[flat_hess[i][j].transpose() for j in range(i)] +
                 [flat_hess[j][i] for j in range(i, len(flat_hess))] for i in range(len(flat_hess))]
    flat_hess = np.vstack(list(map(np.hstack, flat_hess)))
    like0 = mle.like(mle.params_).values
    bump = 1e-3
    for k in range(flat_params.size):
        bumped_flat_params = flat_params.copy()
        bumped_flat_params[k] += bump
        like_bump = mle.like(mle._reshape_params(bumped_flat_params)).values
        der_k = ((like_bump - like0) / bump)
        # print(flat_der[k], "\t", der_k, '\t', np.abs(flat_der[k] - der_k))

    for i in range(flat_params.size):
        bumped_flat_params_i = flat_params.copy()
        bumped_flat_params_i[i] += bump
        like_i = mle.like(mle._reshape_params(bumped_flat_params_i)).values
        for j in range(i + 1):
            bumped_flat_params = flat_params.copy()
            bumped_flat_params[j] += bump
            like_j = mle.like(mle._reshape_params(bumped_flat_params)).values
            bumped_flat_params[i] += bump
            like_ij = mle.like(mle._reshape_params(bumped_flat_params)).values
            hess_ij = (like_ij - like_i - like_j + like0) / bump / bump
            if abs(hess_ij) > 1e-2:
                if abs(flat_hess[i][j] - hess_ij) > 1e-2:
                    print(i, j, '\t', flat_hess[i][j], '\t', hess_ij, '\t', flat_hess[i][j] - hess_ij)

    #s_a, s_b, s_h, s_h1 = mle.std_error()
    # der * der.transpose())[2].values wrong sign
    