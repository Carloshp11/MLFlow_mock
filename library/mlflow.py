import mlflow


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
    mlflow.tracking.MlflowClient().delete_run(run_id=run_id)