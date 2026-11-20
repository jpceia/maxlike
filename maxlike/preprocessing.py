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
        df = observations.groupby(observations.index).agg(**transformations).\
            reindex(pd.MultiIndex.from_product(axis)).fillna(0)
    else:
        axis = sorted(observations.index.unique())    
        if add_axis is not None:
            axis = sorted(set(axis).union(add_axis))
        shape = (len(axis))
        df = observations.groupby(observations.index).\
            agg(**transformations).reindex(axis).fillna(0)
    
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
        agg(**transformations).reindex(new_index).fillna(0)
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


def _columns(cols):
    if isinstance(cols, (list, tuple)):
        return list(cols)
    return [cols]


def prepare_index(df, groups):
    """
    Index arrays for an observation-indexed model (maxlike.func.Take), from
    a DataFrame with one row per observation.

    Parameters
    ----------
    df : pd.DataFrame
        One row per observation.
    groups : dict
        name of an effect -> column, or list of columns, holding its labels.
        Columns listed together share one set of labels, e.g.
        {"team": ["home", "away"], "referee": "referee"}.

    Returns
    -------
    index : dict
        column -> int array with the position of the label of each row.
    labels : dict
        name -> pd.Index of the (sorted) labels: entry k of the parameter is
        the effect of labels[name][k].

    Example
    -------
    >>> index, labels = prepare_index(df, {"team": ["home", "away"]})
    >>> model = Sum(1)
    >>> model.add(Take(index["home"]), 0, 0)
    >>> model.add(-Take(index["away"]), 1, 0)
    >>> mle.add_param(np.zeros(len(labels["team"])), labels=labels["team"])
    """
    index, labels = {}, {}
    for name, cols in groups.items():
        cols = _columns(cols)
        values = pd.concat([df[c] for c in cols], ignore_index=True)
        if values.isna().any():
            raise ValueError("missing labels in column(s) %s" % cols)
        level = pd.Index(values.unique())
        try:
            level = level.sort_values()
        except TypeError:
            pass    # labels that cannot be ordered: order of appearance
        labels[name] = level
        for c in cols:
            index[c] = level.get_indexer(df[c]).astype(np.intp)
    return index, labels


def prepare_observations(df, groups, sums, by=None):
    """
    Same as prepare_index, after adding up the rows with the same labels:
    the likelihoods only need the number of rows (N) and the totals of the
    results for each distinct combination, so repeated observations (two
    teams that met several times, ...) become a single one.

    Parameters
    ----------
    df : pd.DataFrame
        One row per observation.
    groups : dict
        As in prepare_index.
    sums : str, list
        Column(s) to add up (the results, e.g. goals or wins).
    by : str, list (optional)
        Other columns that distinguish observations (covariates, e.g.
        venue); they are returned in data.

    Returns
    -------
    index, labels : as in prepare_index, one entry per distinct combination.
    data : dict
        "N" (number of rows), the totals of the columns in `sums` and the
        values of the columns in `by`, as arrays with one entry per
        distinct combination.
    """
    sums = _columns(sums)
    by = [] if by is None else _columns(by)
    keys = [c for cols in groups.values() for c in _columns(cols)] + by
    grouped = df.groupby(keys, sort=True, observed=True)
    agg = grouped[sums].sum()
    agg.insert(0, "N", grouped.size())
    agg = agg.reset_index()
    index, labels = prepare_index(agg, groups)
    data = {col: agg[col].to_numpy(dtype=float) for col in ["N"] + sums}
    data.update({col: agg[col].to_numpy() for col in by})
    return index, labels, data
