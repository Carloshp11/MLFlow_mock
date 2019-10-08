import argparse
from typing import Iterable

import mlflow
import mlflow.sklearn
import neptune
import numpy as np
# import modin
# import modin.pandas as pd
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from library.etl import temporal_cross_validator
from library.mlflow import burn_first_run
from support_modules.config import MLFlowonfig
from support_modules.misc import join_dicts

neptune.set_project('carlos.heras/MLflow')

parser = argparse.ArgumentParser()
parser.add_argument('-exp_id', '--experiment_id', required=False, type=str)
parser.add_argument('-name', '--run_name', required=False, type=str)
pyargs = parser.parse_args()

start_run_args = {k: v for k, v in pyargs.__dict__.items()
                  if k in ('run_name', 'run_id', 'experiment_id')
                  and v is not None and v.lower() != 'none'}

conf = MLFlowonfig(config_path='conf.yaml')
conf.post_process_dinamic_params()


def evaluate_regression(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {'mse': mean_squared_error(y_test, pred),
            'mae': mean_absolute_error(y_test, pred),
            'msae': np.sqrt(mean_absolute_error(y_test, pred))}
    # 'msle': mean_squared_log_error(y_test, pred)}


def get_best(all_metrics: pd):  # mockup
    metrics = ['mse_mean', 'mse_std',
               'mae_mean', 'mae_std',
               'msae_mean', 'msae_std']

    best = all_metrics.iloc[0]
    best_metrics = best[metrics].to_dict()
    best_hyperparameters = best.drop(metrics).to_dict()

    return best_metrics, best_hyperparameters


def workflow(df, standard_hyperparameters: dict, model_iterator_: Iterable):
    # ETL
    predictive_features = [_ for _ in df.columns if _ not in (conf.label_col, conf.temporal_col)]  # No ETL
    print('ETL hyperparameters: {}'.format(standard_hyperparameters))
    # Model Optimization
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
