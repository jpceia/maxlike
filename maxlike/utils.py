import numpy as np
import pandas as pd


def prepare_series(observations, transformations=None):
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

    Returns
    -------
    res : dict
        resulting ndarrays after applying the transformations on the
        observations
    axis : list[list]
        feature index names
    """
    if transformations is None:
        transformations = {"N": np.size}

    if isinstance(observations.index, pd.MultiIndex):
        axis = tuple((level.sort_values()
                      for level in observations.index.levels))
        shape = tuple((len(a) for a in axis))
        df = observations.groupby(observations.index).\
            agg(transformations.values()).\
            rename(columns={transf.__name__: name
                            for name, transf in transformations.items()}).\
            reindex(pd.MultiIndex.from_product(axis)).fillna(0)
    else:
        axis = observations.index.sort_values()
        shape = (axis.size)
        df = observations.groupby(axis).agg(transformations.values()).\
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
            name = level.name
            if name in add_axis:
                axis[k] = sorted(set(axis[k]).union(add_axis[name]))
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
        columns=pd.Index(np.arange(size),
            name=name))
