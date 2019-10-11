import argparse
from typing import Iterable, Tuple
import lightgbm
import mlflow
import mlflow.sklearn
import neptune
# import modin
# import modin.pandas as pd
import pandas as pd

from library.code_patterns import AttDict
from library.etl import temporal_cross_validator
from library.mlflow import burn_first_run, manage_runs
from support_modules.arg_validators import valid_mode
from support_modules.config import MLFlowonfig
from support_modules.misc import join_dicts

# neptune.set_project('carlos.heras/MLflow')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', required=True, type=valid_mode)
parser.add_argument('-exp_id', '--experiment_id', required=False, type=str)
parser.add_argument('-name', '--run_name', required=False, type=str)
pyargs = parser.parse_args()

train = pyargs.mode == 'train'
start_run_args = {k: v for k, v in pyargs.__dict__.items()
                  if k in ('run_name', 'run_id', 'experiment_id')
                  and v is not None and v.lower() != 'none'}

conf = MLFlowonfig(config_path='conf.yaml')
conf.post_process_dinamic_params()


def get_best(all_metrics: pd):  # mockup
    metrics = ['mse_mean', 'mse_std',
               'mae_mean', 'mae_std',
               'msae_mean', 'msae_std']

    best = all_metrics.iloc[0]
    best_metrics = {k: v for k, v in best[metrics].to_dict().items() if not pd.isna(v)}
    best_hyperparameters = best.drop(metrics).to_dict()

    return best_metrics, best_hyperparameters


def optimize_model(df: pd, predictive_features: list, config: AttDict) -> Tuple[any, dict, dict]:
    assert 'hyperparameters' in config.keys()

    all_metrics = pd.DataFrame()
    # noinspection PyCallingNonCallable
    for model_class, model_name, hyperparameters_combination in config.hyperparameters.models_iterator():
        model = model_class(**hyperparameters_combination)
        metrics = temporal_cross_validator(model=model,
                                           df=df[[config.temporal_col, config.label_col] + predictive_features],
                                           label_col=config.label_col,
                                           temporal_col=config.temporal_col,
                                           n_folds=5,
                                           n_folds_training=2,
                                           min_validation_perc=0.2)
        all_metrics = all_metrics.append(pd.DataFrame(data=join_dicts({'model_name': model_name},
                                                                      hyperparameters_combination,
                                                                      metrics),
                                                      index=[0]), ignore_index=True, sort=False)  #  Append with different columns
    best_metrics, best_hyperparameters = get_best(all_metrics)
    # noinspection PyUnboundLocalVariable
    fitted_model = model.fit(df[predictive_features], df[config.label_col])  # We shall add test df
    return fitted_model, best_metrics, best_hyperparameters


def workflow(df, etl_hyperparameters: dict, flow):
    global train
    # ETL
    predictive_features = [_ for _ in df.columns if _ not in (conf.label_col, conf.temporal_col)]  # No ETL

    # Model Optimization
    if train:
        fitted_model, best_metrics, best_hyperparameters = optimize_model(df,
                                                                          predictive_features=predictive_features,
                                                                          config=conf)
    else:
        pass

    # MLFlow logging
    mlflow.log_params(best_hyperparameters)
    mlflow.log_metrics(best_metrics)
    mlflow.sklearn.log_model(fitted_model, "model")


if __name__ == "__main__":
    # Load data
    dataset = pd.read_csv('local_storage/winequality-red_timestamp.csv', sep=';')
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], infer_datetime_format=True)
    burn_first_run()

    # Lo que no dependa de ningún hiperparámetro debe quedarse fuera de este bucle
    for flow_, etl_hyperparameters_ in manage_runs(conf.hyperparameters.standard_grid):
        print('etl_hyperparameters_\n', etl_hyperparameters_)
        workflow(dataset, etl_hyperparameters_, flow_)  # Has no validation set reserved

    # for etl_hyperparameters_, model_iterator in conf.hyperparameters:
    #     workflow(dataset, etl_hyperparameters_, model_iterator)  # Has no validation set reserved

    print('THE END')
