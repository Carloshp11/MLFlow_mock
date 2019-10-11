import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_regression(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {'mse': mean_squared_error(y_test, pred),
            'mae': mean_absolute_error(y_test, pred),
            'msae': np.sqrt(mean_absolute_error(y_test, pred))}
    # 'msle': mean_squared_log_error(y_test, pred)}