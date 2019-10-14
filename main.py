import argparse
from typing import Tuple, Iterable
import mlflow
import mlflow.sklearn
import neptune
# import modin
# import modin.pandas as pd
import pandas as pd

from library.code_patterns import AttDict
from library.config import ConfigBase
from library.etl import temporal_cross_validator, MixedParameterGrid
from library.mlflow import burn_first_run, manage_runs, parse_run_params
from support_modules.arg_validators import valid_mode
from support_modules.config import MLFlowConfig
from support_modules.misc import join_dicts

# neptune.set_project('carlos.heras/MLflow')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', required=True, type=valid_mode)
parser.add_argument('-exp_id', '--experiment_id', required=False, type=str)
parser.add_argument('-name', '--run_name', required=False, type=str)
parser.add_argument('-run', '--run_id', required=False, type=str)
pyargs = parser.parse_args()

train = pyargs.mode == 'train'
start_run_args = {k: v for k, v in pyargs.__dict__.items()
                  if k in ('run_name', 'experiment_id')
                  and v is not None and v.lower() != 'none'}

conf = ConfigBase(config_path='conf.yaml')


def get_best(all_metrics: pd):  # mockup
    metrics = ['mse_mean', 'mse_std',
               'mae_mean', 'mae_std',
               'msae_mean', 'msae_std']

    best = all_metrics.iloc[0]
    best_metrics = {k: v for k, v in best[metrics].to_dict().items() if not pd.isna(v)}
    best_hyperparameters = best.drop(metrics).to_dict()

    return best_metrics, best_hyperparameters


def optimize_model(df: pd, predictive_features: list, models_iterator: Iterable, config: AttDict) -> Tuple[any, dict, dict]:
    assert 'hyperparameters' in config.keys()

    all_metrics = pd.DataFrame()
    # noinspection PyCallingNonCallable
    for model_class, model_name, hyperparameters_combination in models_iterator:
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


class Workflow:

    def __init__(self, config: AttDict, mode: str):
        self.conf = config
        self.conf.hyperparameters = MixedParameterGrid(config.hyperparameters)
        self.train = mode == 'train'
        self.MLFlowClient = mlflow.tracking.MlflowClient()

    def manage_runs(self):
        return manage_runs(self.conf.hyperparameters.standard_grid)

    def _models_iterator_(self):
        return self.conf.hyperparameters.models_iterator()

    def etl(self, df: pd, etl_hyperparameters: dict):
        print('etl_hyperparameters\n', etl_hyperparameters)
        self.predictive_features = [_ for _ in df.columns if _ not in (conf.label_col, conf.temporal_col)]
        return df

    def fit(self, df: pd, etl_hyperparameters: dict, flow: mlflow.tracking.fluent.ActiveRun):
        df = self.etl(df, etl_hyperparameters)

        fitted_model, best_metrics, best_hyperparameters = optimize_model(df,
                                                                          predictive_features=self.predictive_features,
                                                                          models_iterator=self._models_iterator_(),
                                                                          config=self.conf)

        # MLFlow logging
        mlflow.log_params(best_hyperparameters)
        mlflow.log_metrics({'etl_'+k: v for k, v in best_metrics.items()})
        mlflow.sklearn.log_model(fitted_model, 'model')

    def predict(self, df: pd, run_id: str):
        etl_hyperparameters, _ = parse_run_params(self.MLFlowClient, run_id)
        df = self.etl(df, etl_hyperparameters)

        fitted_model = mlflow.sklearn.load_model(self.MLFlowClient.download_artifacts(run_id, 'model'))
        df['predction'] = fitted_model.predict(df[self.predictive_features])
        return df


if __name__ == "__main__":
    workflow = Workflow(config=conf, mode=pyargs.mode)

    # Load data
    dataset = pd.read_csv('local_storage/winequality-red_timestamp.csv', sep=';')
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], infer_datetime_format=True)

    # Whatever does not depend upon any hyperparameter shall go here
    pass

    if train:
        burn_first_run()
        for flow_, etl_hyperparameters_ in manage_runs(conf.hyperparameters.standard_grid):
            workflow.fit(dataset, etl_hyperparameters_, flow_)  # Has no validation set reserved
    else:
        predicted_df = workflow.predict(dataset, pyargs.run_id)
        print('predicted_df')
        print(predicted_df)
        print('predicted_df AGAIN')
        print(predicted_df)

    print('THE END')
