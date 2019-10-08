import math

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from main import evaluate_regression


def temporal_cross_splitter(X: pd, y: pd.Series, temporal_col: str, n_folds: int = 3, n_folds_training: int = 2,
                            min_validation_perc: float = 0.33):  # We could also implement supersets training folds
    assert ptypes.is_datetime64_any_dtype(
        X[temporal_col]), 'The temporal column {} must be datetime but its dtype is {}'.format(temporal_col,
                                                                                               X.dtypes[temporal_col])
    min_n_folds_test = math.ceil(n_folds * min_validation_perc)
    n_steps = n_folds - n_folds_training - min_n_folds_test + 1

    tmp_index = pd.cut(X[temporal_col], n_folds)
    folds = tmp_index.drop_duplicates().sort_values()

    for i in range(n_steps):
        # print('Starting validation of train_folds {}-{} against test_folds {}-{}'
        #      .format(i+1,i+n_folds_training,
        #              i+n_folds_training+1, len(folds)))
        folds_train = folds[i:i + n_folds_training].to_list()
        folds_test = folds[i + n_folds_training:].to_list()

        y_train = y.loc[tmp_index.isin(folds_train)]
        y_test = y.loc[tmp_index.isin(folds_test)]
        X_train = X.loc[y_train.index]
        X_test = X.loc[y_test.index]

        yield X_train, X_test, y_train, y_test


def temporal_cross_validator(model, df: pd, label_col: str = 'label', temporal_col: str = None, n_folds: int = 3,
                             n_folds_training: int = 2, min_validation_perc: float = 0.33):
    list_of_metrics = None
    for X_train, X_test, y_train, y_test in temporal_cross_splitter(df.drop(label_col, axis=1),
                                                                    df[label_col],
                                                                    temporal_col=temporal_col,
                                                                    n_folds=n_folds,
                                                                    n_folds_training=n_folds_training,
                                                                    min_validation_perc=min_validation_perc):
        model.fit(X_train.drop(temporal_col, axis=1), y_train)
        metrics = evaluate_regression(model, X_test.drop(temporal_col, axis=1), y_test)
        if list_of_metrics:
            list_of_metrics = {k: list_of_metrics[k] + [v]
                               for k, v in metrics.items()}
        else:
            list_of_metrics = {k: [v]
                               for k, v in metrics.items()}
    cv_metrics = {k + '_mean': np.mean(v)
                  for k, v in list_of_metrics.items()}
    std_metrics = {k + '_std': np.std(v)
                   for k, v in list_of_metrics.items()}
    cv_metrics.update(std_metrics)
    return cv_metrics