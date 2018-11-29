import numpy as np


def proba(odds, tol=1e-15, max_steps=50, method="basic"):
    
    x = 1 / odds
    N = x.shape[-1]

    if method.lower() == "shin":
        if odds.ndim == 1:
            x2vec = x * x / x.sum()
            z = 0
            for k in range(max_steps):
                sqvec = np.sqrt(z * z + 4 * (1 - z) * x2vec)
                foo = sqvec.sum() - (N - 2) * z - 2
                jac = ((z - 2 * x2vec) / sqvec).sum() - (N - 2)
                z -= foo / jac
                if np.abs(foo) < tol:
                    break
            else:
                raise ValueError

            return (sqvec - z) / (2 * (1 - z))

        if odds.ndim == 2:
            x2vec = x * x / x.sum(1)[:, None]
            z = np.zeros(x.shape[0])
            for k in range(max_steps):
                sqvec = np.sqrt((z * z)[:, None] + 4 * (1 - z)[:, None] * x2vec)
                foo = sqvec.sum(1) - (N - 2) * z - 2
                jac = ((z[:, None] - 2 * x2vec) / sqvec).sum(1) - (N - 2)
                z -= foo / jac
                if np.abs(foo).max() < tol:
                    break
            else:
                raise ValueError

            return (sqvec - z[:, None]) / (2 * (1 - z[:, None]))

    elif method.lower() == "basic":
        if odds.ndim == 1:
            return x / x.sum()
        elif odds.ndim == 2:
            return x / x.sum(-1)[:, None]

    raise ValueError
