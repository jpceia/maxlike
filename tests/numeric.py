import numpy as np


def num_grad(mle, params, e=1e-6):

    flat_params = np.concatenate([
        np.asarray(p)[~p_.mask] for p, p_ in zip(params, mle.params)])
    y = mle.model(params)
    like_0 = float(mle.like(params, y).values)
    num_grad = np.zeros_like(flat_params)

    for i in range(len(flat_params)):
        bump = np.zeros_like(flat_params)
        bump[i] = e
        bumped_params = mle._reshape_params(flat_params + bump)
        y_bumped = mle.model(bumped_params)
        num_grad[i] = (float(mle.like(bumped_params, y_bumped).values) - like_0) / e

    return mle._reshape_array(num_grad)


def num_hess(mle, params, e=1e-6):

    flat_params = np.concatenate([
        np.asarray(p)[~p_.mask] for p, p_ in zip(params, mle.params)])
    y = mle.model(params)
    like_0 = mle.like(params, y).values
    like_i = np.zeros_like(flat_params)
    n = len(flat_params)

    for i in range(n):
        bump = np.zeros_like(flat_params)
        bump[i] = e
        bumped_params = mle._reshape_params(flat_params + bump)
        y_bumped = mle.model(bumped_params)
        like_i[i] = mle.like(bumped_params, y_bumped).values
    num_hess = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            bump = np.zeros_like(flat_params)
            bump[i] = e
            bump[j] += e
            bumped_params = mle._reshape_params(flat_params + bump)
            y_bumped = mle.model(bumped_params)
            num_hess[i][j] = (mle.like(bumped_params, y_bumped).values
                              + like_0 - like_i[i] - like_i[j]) / e / e
    return mle._reshape_matrix(np.array(num_hess))
