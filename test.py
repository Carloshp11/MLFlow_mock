import mlflow

with mlflow.start_run(nested=False) as flow1:
    mlflow.log_param('level', 1)
    with mlflow.start_run(nested=True) as flow2:
        mlflow.log_param('level', 2)
        with mlflow.start_run(nested=True) as flow3:
            mlflow.log_param('level', 3)
            mlflow.log_param('some_param', 'Something')
