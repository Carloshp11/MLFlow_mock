from typing import Iterable

import mlflow
from sklearn.model_selection import ParameterGrid

from support_modules.misc import join_dicts


def get_commit(active_run):
    return active_run.data.tags['mlflow.source.git.commit']


def set_commit(active_run, commit):
    active_run._data._tags['mlflow.source.git.commit'] = commit


def burn_first_run():
    """
    This is required if you plan on using the run_name mlflow param and launch several runs on a single script.
    If you do not run this function, the first run will have the run_name as a param instead of a main attribute on the ui.
    :return: None
    """
    with mlflow.start_run() as flow:
        run_id = flow.info.run_id
        mlflow.tracking.MlflowClient().set_terminated(run_id=run_id)
    mlflow.tracking.MlflowClient().delete_run(run_id=run_id)


def manage_runs(params_grid: dict, _deep_values: dict = None, **start_run_args) -> Iterable:
    param_name = list(params_grid.keys())[0]
    first_param = {k: v for k, v in params_grid.items() if k == param_name}
    other_params = {k: v for k, v in params_grid.items() if k != param_name}

    param_values = first_param[param_name]
    for param_value in param_values:
        next_iteration = {param_name: param_value}
        next_deep_values = join_dicts(_deep_values, next_iteration) if _deep_values else next_iteration

        with mlflow.start_run(**start_run_args, nested=True) as flow:  # Only the deepest flow is returned by the iterator
            mlflow.log_param('etl_'+param_name, param_value)
            if other_params:  #  There are more nested parameters to combine
                yield from manage_runs(other_params, _deep_values=next_deep_values, **start_run_args)
            else:
                yield flow, next_deep_values


def parse_run_params(MLFlowClient: mlflow.tracking.MlflowClient, run_id: str):
    run_info = MLFlowClient.get_run(run_id)
    parent_run_info = MLFlowClient.get_run(run_info.data.tags['mlflow.parentRunId'])
    run_hyperparameters = join_dicts(parent_run_info.data.params, run_info.data.params)
    etl_hyperparameters = {k[4:]: v for k, v in run_hyperparameters.items() if k.startswith('etl_')}
    model_hyperparameters = {k: v for k, v in run_hyperparameters.items() if not k.startswith('etl_')}
    return etl_hyperparameters, model_hyperparameters
