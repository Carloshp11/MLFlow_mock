import mlflow
import modin
import modin.pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from lightgbm.sklearn import LGBMClassifier


def ETL(df: modin.pandas, test_size: float) -> modin.pandas:
    X = df.drop('quality')
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def workflow(df: modin.pandas, test_size: float, boosting_type: str, max_depth: int, learning_rate: float):
    with mlflow.start_run():
        # ETL
        X_train, X_test, y_train, y_test = ETL(df, test_size=test_size)





if __name__ == "__main__":
    # Load data
    dataset = pd.read_csv('local_storage/winequality-red.csv', sep=';')

    grid = ParameterGrid({
        'test_size': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        'boosting_type': ('gbdt', 'dart', 'goss', 'rf'),
        'max_depth': range(3, 12, 2),
        'learning_rate': (0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.04, 0.01)
    })

    for hyperparameters in grid:
        workflow(dataset, **hyperparameters)

print('THE END')