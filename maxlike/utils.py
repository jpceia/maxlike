import pandas as pd


def prepare_series(observations, transformations):
    if isinstance(observations.index, pd.MultiIndex):
        axis = [level.sort_values() for level in observations.index.levels]
        shape = map(len, axis)
        df = observations.groupby(observations.index).agg(transformations).\
            reindex(pd.MultiIndex.from_product(axis)).fillna(0)
    else:
        axis = observations.index.sort_values()
        shape = (axis.size)
        df = observations.groupby(axis).agg(transformations).reindex(axis).\
            fillna(0)
    res = {k: df[k].values.reshape(shape) for k in transformations.keys()}
    return res, axis


def prepare_dataframe(df, features_cols, result_col,
                      transformations, weight_col=None):
    raise NotImplementedError
