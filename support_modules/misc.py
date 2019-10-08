import pickle

import mlflow
from sklearn.model_selection import ParameterGrid
from typing import Iterable, Union, Tuple, List

import boto3
import pandas as pd
import s3fs

from library.debug import ProgressBar


def date_to_integer(dt_time):
    return 10000 * dt_time.year + 100 * dt_time.month + dt_time.day


def get_key(path):
    bucket = get_bucket(path)
    return path.split(bucket)[-1][1:]


def get_bucket(path):
    return path.split('//')[1].split('/')[0]


def save_model_to_s3(model_, path):
    fs = s3fs.S3FileSystem(anon=False)
    with fs.open(path, 'wb') as f:
        pickle.dump(model_, f)


def load_model_from_s3(path, access_key, secret_key):
    s3client = boto3.client('s3',
                            aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key)

    response = s3client.get_object(Bucket=get_bucket(path), Key=get_key(path))
    body = response['Body'].read()
    data = pickle.loads(body)
    return data


def df_crossjoin(df1, df2, **kwargs):
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1
    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))
    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)
    return res


class MixedParameterGrid:
    """
    This class is designed to allow users to define a complex hyperparameter grid on a .yaml or other plain text conf file.
    This grid can, most importantly, hold 'standard' or non-model hyperparameters and then a dictionary of 'in-model' hyperparameters for each of them.
    In order for the class to work properly, the input dict must contain a 'models' key which must be a dict itself. This models dict
    must have as keys valid import paths to the model objects (example 'lightgbm.sklearn.LGBMRegressor') and as values, the hyperparameters values as a list.

    Example
    =======


    """

    def __init__(self, mixed_param_grid: dict):
        assert 'models' in mixed_param_grid.keys(), 'A MixedParameterGrid dictionary must have a \'models\' key. Else, use a sklearn.model_selection.ParameterGrid object instead'
        self.standard_grid = {k: v for k, v in mixed_param_grid.items() if k.lower() != 'models'}
        self.models_grid = mixed_param_grid['models']

    def __iter__(self) -> Tuple[dict, Iterable]:
        for standard_hyperparameters in ParameterGrid(self.standard_grid):
            yield standard_hyperparameters, self._models_iterator_

    def _models_iterator_(self) -> Tuple[any, str, dict]:
        for model_name, model_hyperparameters in self.models_grid.items():
            assert isinstance(model_hyperparameters, dict)
            for k, v in model_hyperparameters.items():
                if not isinstance(v, (list, tuple)):
                    model_hyperparameters[k] = [v]

            progess_bar = ProgressBar(total=len(ParameterGrid(model_hyperparameters)),
                                      prefix=model_name, print_each=3)

            model = self._import_from_string_spec_(model_name)
            for hyperparameters_combination in ParameterGrid(model_hyperparameters):
                progess_bar.print()
                yield model, model_name, hyperparameters_combination

    @staticmethod
    def _import_from_string_spec_(spec):
        import importlib
        from_ = '.'.join(spec.split('.')[:-1])
        import_ = spec.split('.')[-1]
        module = importlib.import_module(from_, package=import_)
        return getattr(module, import_)


def join_dicts(*dicts) -> dict:
    def join_dict_with_list(dict1: dict, l: Tuple[dict]):
        return join_dict_with_list({**dict1, **l[0]}, l[1:]) if len(l) > 0 else dict1

    assert len([k for dict_ in dicts for k in dict_.keys()]) == len(set([k for dict_ in dicts for k in dict_.keys()])), \
        'dicts have one or more duplicated keys. Join not possible'
    return join_dict_with_list(dicts[0], dicts[1:])
