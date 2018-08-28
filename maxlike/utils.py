import numpy as np
import pandas as pd


def prepare_series(observations, transformations=None, add_axis=None):
    """
    Extracts the relevant parameters from a Series of observations to feed the
    maxlike object.

    Parameters
    ----------
    observations : pd.Series
        sequence of observations, the index correspond to the features and the
        values to the target values.
    transformations : dict
        (named) list of transformations to apply to observations labels,
        grouped by index
    add_axis: dict, list
        Different axis to use. Dictionary if the observations have multiIndex,
        list if vanilla index

    Returns
    -------
    res : dict
        resulting ndarrays after applying the transformations on the
        observations
    axis : list[list]
        feature index names
    """
    assert isinstance(observations, pd.Series)

    if transformations is None:
        transformations = {"N": np.sum}

    if isinstance(observations.index, pd.MultiIndex):
        axis = [level.sort_values() for level in observations.index.levels]
        if add_axis is not None:
            for k, level in enumerate(observations.index.levels):
                if level.name in add_axis:
                    axis[k] = sorted(set(axis[k]).union(add_axis[level.name]))
        shape = tuple((len(a) for a in axis))
        df = observations.groupby(observations.index).\
            agg(transformations.values()).\
            rename(columns={transf.__name__: name
                            for name, transf in transformations.items()}).\
            reindex(pd.MultiIndex.from_product(axis)).fillna(0)
    else:
        axis = sorted(observations.index.unique())    
        if add_axis is not None:
            axis = sorted(set(axis).union(add_axis))
        shape = (len(axis))
        df = observations.groupby(observations.index).\
            agg(transformations.values()).\
            rename(columns={transf.__name__: name
                            for name, transf in transformations.items()}).\
            reindex(axis).fillna(0)
    res = {k: df[k].values.reshape(shape) for k in transformations.keys()}
    return res, axis


def prepare_dataframe(df, weight_col, result_col, transformations,
                      add_axis=None):
    axis = [level.sort_values() for level in df.index.levels]
    if add_axis is not None:
        for k, level in enumerate(df.index.levels):
            if level.name in add_axis:
                axis[k] = sorted(set(axis[k]).union(add_axis[level.name]))
    shape = tuple((len(a) for a in axis))
    new_index = pd.MultiIndex.from_product(axis)
    w = df[weight_col].to_frame('N').groupby(df.index).sum().\
        reindex(new_index).fillna(0)
    df = (df[result_col] * df[weight_col]).groupby(df.index).\
        agg(transformations.values()).\
        rename(columns={transf.__name__: name
                        for name, transf in transformations.items()}).\
        reindex(new_index).fillna(0)
    res = {k: df[k].values.reshape(shape) for k in transformations.keys()}
    res['N'] = w.values.reshape(shape)
    return res, axis


def df_count(series, size, name=None):
    if not name:
        name = 'count'
    X = series.values[:, None] == np.arange(size - 1)[None, :]
    return pd.DataFrame(
        np.insert(X, -1, X.sum(-1) == False, 1),
        index=series.index,
        columns=pd.Index(np.arange(size), name=name))


def num_grad(mle, params, e=1e-6):

    flat_params = np.concatenate([
        np.asarray(p)[~p_.mask] for p, p_ in zip(params, mle.params)])
    like_0 = float(mle.like(params).values)
    num_grad = np.zeros_like(flat_params)

    for i in range(len(flat_params)):
        bump = np.zeros_like(flat_params)
        bump[i] = e
        bumped_params = mle._reshape_params(flat_params + bump)
        num_grad[i] = (float(mle.like(bumped_params).values) - like_0) / e

    return mle._reshape_array(num_grad)


def num_hess(mle, params, e=1e-6):

    flat_params = np.concatenate([
        np.asarray(p)[~p_.mask] for p, p_ in zip(params, mle.params)])
    like_0 = mle.like(params).values
    like_i = np.zeros_like(flat_params)
    n = len(flat_params)

    for i in range(n):
        bump = np.zeros_like(flat_params)
        bump[i] = e
        bumped_params = mle._reshape_params(flat_params + bump)
        like_i[i] = mle.like(bumped_params).values
    num_hess = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            bump = np.zeros_like(flat_params)
            bump[i] = e
            bump[j] += e
            bumped_params = mle._reshape_params(flat_params + bump)
            num_hess[i][j] = (mle.like(bumped_params).values
                              + like_0 - like_i[i] - like_i[j]) / e / e
    return mle._reshape_matrix(np.array(num_hess))
