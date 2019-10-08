import argparse

import math
import numpy as np
import pandas.api.types as ptypes
from typing import List, Union, Iterable

import mlflow
import mlflow.sklearn
# import modin
# import modin.pandas as pd
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm.sklearn import LGBMRegressor

from library.config import ConfigBase
from library.debug import ProgressBar
from support_modules.config import MLFlowonfig
from support_modules.misc import MixedParameterGrid, join_dicts

parser = argparse.ArgumentParser()
parser.add_argument('-exp_id', '--experiment_id', required=False, type=str)
parser.add_argument('-name', '--run_name', required=False, type=str)
pyargs = parser.parse_args()

start_run_args = {k: v for k, v in pyargs.__dict__.items()
                  if k in ('run_name', 'run_id', 'experiment_id')
                  and v is not None and v.lower() != 'none'}

conf = MLFlowonfig(config_path='conf.yaml')
conf.post_process_dinamic_params()


def get_non_gitignore_contents_in_dir(path: str, gitignore=None, extra_ignore=None, depth=0, max_recursion=99) \
        -> List[str]:
    import os
    from fnmatch import fnmatch

    def filtered(names: List[str], patterns) -> List[str]:
        return [name for name in names
                if not any([fnmatch(os.path.basename(name), pattern)
                            for pattern in patterns])]

    if extra_ignore is None:
        extra_ignore = []

    if gitignore is None:
        try:
            gitignore = file_to_list(os.path.join(path, '.gitignore'))
        except FileNotFoundError:
            gitignore = []
    gitignore += extra_ignore

    all_contents = [os.path.join(path, c) for c in os.listdir(path) if not os.path.islink(os.path.join(path, c))]
    files = [f for f in all_contents if os.path.isfile(f)]
    folders = [f for f in all_contents if os.path.isdir(f)]

    contents = filtered(files, gitignore)

    if depth < max_recursion:
        for folder in filtered(folders, gitignore):
            contents += get_non_gitignore_contents_in_dir(folder,
                                                          gitignore=gitignore,
                                                          depth=depth + 1,
                                                          max_recursion=max_recursion)

    # for dirname, _, filenames in os.walk(path):
    #     for filename in filenames:
    #         if not any([fnmatch(filename, pattern)
    #                     for pattern in gitignore]):
    #             contents.append(filename)
    return contents


def file_to_list(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [line.rstrip('\n') for line in f.readlines()]


# def ETL(df: modin.pandas, test_size: float) -> modin.pandas:
#     X = df.drop('quality')
#     y = df['quality']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#     return X_train, X_test, y_train, y_test


def evaluate_regression(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {'mse': mean_squared_error(y_test, pred),
            'mae': mean_absolute_error(y_test, pred),
            'msae': np.sqrt(mean_absolute_error(y_test, pred))}
    # 'msle': mean_squared_log_error(y_test, pred)}


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


def get_best(all_metrics: pd):  # mockup
    metrics = ['mse_mean', 'mse_std',
               'mae_mean', 'mae_std',
               'msae_mean', 'msae_std']

    best = all_metrics.iloc[0]
    best_metrics = best[metrics].to_dict()
    best_hyperparameters = best.drop(metrics).to_dict()

    return best_metrics, best_hyperparameters


# def get_commit(active_run):
#     return active_run.data.tags['mlflow.source.git.commit']
#
#
# def set_commit(active_run, commit):
#     active_run._data._tags['mlflow.source.git.commit'] = commit


def burn_first_run():
    """
    This is required if you plan on using the run_name mlflow param and launch several runs on a single script.
    If you do not run this function, the first run will have the run_name as a param instead of a main attribute on the ui.
    :return: None
    """
    with mlflow.start_run() as flow:
        run_id = flow.info.run_id
    mlflow.tracking.MlflowClient().delete_run(run_id=run_id)


def workflow(df, standard_hyperparameters: dict, model_iterator_: Iterable):
    # ETL
    predictive_features = [_ for _ in df.columns if _ not in (conf.label_col, conf.temporal_col)]  # No ETL
    print('ETL hyperparameters: {}'.format(standard_hyperparameters))
    # Model Optimization
    print('start_run_args')
    print(start_run_args)
    with mlflow.start_run(**start_run_args) as flow:
        # git_commit = get_commit(flow)
        # set_commit(flow, '666')
        all_metrics = pd.DataFrame()
        # noinspection PyCallingNonCallable
        for model_class, model_name, hyperparameters_combination in model_iterator_():
            model = model_class(**hyperparameters_combination)
            metrics = temporal_cross_validator(model=model,
                                               df=df[[conf.temporal_col, conf.label_col] + predictive_features],
                                               label_col=conf.label_col,
                                               temporal_col=conf.temporal_col,
                                               n_folds=5,
                                               n_folds_training=2,
                                               min_validation_perc=0.2)
            all_metrics = all_metrics.append(pd.DataFrame(data=join_dicts({'model_name': model_name},
                                                                          hyperparameters_combination,
                                                                          metrics),
                                                          index=[0]), ignore_index=True)
        best_metrics, best_hyperparameters = get_best(all_metrics)
        fitted_model = model.fit(df[predictive_features], df[conf.label_col])  # We shall add test df

        # MLFlow logging
        mlflow.log_params(standard_hyperparameters)
        mlflow.log_params(best_hyperparameters)
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(fitted_model, "model")


if __name__ == "__main__":
    # get_non_gitignore_contents_in_dir('/Users/chp11/projects/MLFlow_mockup', ['mlruns'])
    # Load data
    dataset = pd.read_csv('local_storage/winequality-red_timestamp.csv', sep=';')
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], infer_datetime_format=True)
    burn_first_run()

    # Lo que no dependa de ningún hiperparámetro debe quedarse fuera de este bucle
    for standard_hyperparameters_, model_iterator in conf.hyperparameters:
        workflow(dataset, standard_hyperparameters_, model_iterator)  # Has no validation set reserved

    print('THE END')
