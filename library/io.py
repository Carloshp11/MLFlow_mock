import os
import subprocess
from copy import copy

import yaml
import pickle
import shutil
import s3fs

import pandas as pd
import pyarrow
from pyarrow import parquet as pq

from typing import List, Union, Tuple
from library.debug import conditional_print as print


def load_yaml(file: str) -> dict:
    """
    Carga en memoria un fichero .yml y lo devuelve en forma de diccionario.
    :param file: Ruta hasta el fichero que se desea cargar en memoria.
    :return: Contenido del fichero en forma de diccionario.
    """
    with open(file, 'r') as stream:
        configuration = yaml.load(stream, Loader=yaml.SafeLoader)
    return configuration


def pretty_dict(d: dict, indent=0, keys_only=True):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if not keys_only:
            if isinstance(value, dict):
                pretty_dict(value, indent + 1, keys_only=keys_only)
            else:
                print('\t' * (indent + 1) + str(value))


def remove_if_exists(path: str):
    if os.path.exists(path):
        if os.path.isdir(path):
            if not os.listdir(path):  # Then it's empty
                os.rmdir(path)
            else:
                shutil.rmtree(path)
        else:
            os.remove(path)


def execute_bash(args: List[str], continue_on_error: bool = False) -> Tuple:
    from subprocess import Popen, PIPE

    # print('executing bash: {}'.format(args))
    p = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    rc = p.returncode
    if not continue_on_error and rc != 0:
        raise Exception('execute_bash {} got the following error:'.format(args) + str(err))
    if args[0] == 'ls':
        output = str(output).split('\\n')
    return output, err, rc


def ensure_path(path: str, del_if_exists: bool = True, include_last: bool = False):
    """
    Makes sure a path exists, creating all subfolders required otherwise.
    :param path: The path to ensure.
    :param del_if_exists: If True and there is already a file/folder in path, delete it.
    :return: None.
    """
    abs_path = '/' if path[0] == '/' else ''
    folders = list(enumerate([folder for folder in path.split('/') if folder != '']))
    cummulative_paths = [abs_path + os.path.join(*[folder for i, folder in folders if i <= step])
                         for step in range(0, len(folders))]
    iteration = 0
    total_iterations = len(cummulative_paths)
    for cummulative in cummulative_paths:
        # print('ensure', cummulative)
        iteration += 1
        if iteration < (total_iterations + int(include_last)):  # Not the final path yet
            output, err, rc = execute_bash(['ls', cummulative], continue_on_error=True)
            if rc == 0:
                pass
                # if not os.path.isdiative):
            #                 #     raise ValueError('You can\'t ensure a path containing a file in between.\n'
            #                 #                      'File exir(cummulsting at path "{}"'.format(cummulative))
            else:
                execute_bash(['mkdir', cummulative])
                # os.makedirs(cummulative)
        elif del_if_exists:
            remove_if_exists(cummulative)


# def aws_s3_cp_folder(from_: str, to: str) -> None:
#     aws_s3_cp(from_, to, recursive=True)
#
#
# def aws_s3_cp(from_: str, to: str, recursive: bool = False) -> None:
#     ensure_path(to)
#     args = ['aws', 's3', 'cp', '--recursive', from_, to]
#     if not recursive:
#         args.remove('--recursive')
#     print(args)
#     subprocess.run(args)


def save_object_to_s3(model_: object, path: str) -> None:
    fs = s3fs.S3FileSystem(anon=False)
    with fs.open(path, 'wb') as f:
        pickle.dump(model_, f)
    print('Storing to s3: ', path, verbosity=1)


# noinspection PyArgumentList
def write_to_s3_parquet(s3_: Union[s3fs.S3FileSystem, None], df: pd.DataFrame = None, path: str = None,
                        partition_cols: List[str] = None):
    assert df is not None
    df = pyarrow.Table.from_pandas(df)
    df = df.drop([c for c in df.column_names if '__index_level_' in c])
    print('Writing parquet to {}'.format(path))
    pq.write_to_dataset(table=df, root_path=path, filesystem=s3_, partition_cols=partition_cols, preserve_index=False)


def read_parquet(fs: Union[s3fs.S3FileSystem, None],
                 path: str = None,
                 fallback_path: str = None,
                 columns: List[str] = None,
                 partition_filters: Union[List[Tuple[str]], None] = None,
                 non_partition_filters: Union[List[Tuple], List[List[Tuple]], None] = None) -> pd.DataFrame:
    """
    Read parquet file to pandas from S3. Accepts partition filters without overhead and automatically syncronizes to local
    filesystem prior to read if s3 is None.
    :param fs: Filesystem. s3fs.S3FileSystem instance or None if executing on local.
    :param path: path to the parquet folder.
    :param fallback_path: In case you perform a local execution and the path does not exist on your machine, fall back
                          s3 path from which it will be copied.
    :param columns: List[str]
        Names of columns to read from the file
    :param partition_filters: List[Tuple[str]] or None (default)
        One list element for filter. Each tuple contains 1) the column to filter for
        and 2) the value to filter on. List element order matters.
    :param non_partition_filters: List[Tuple] or List[List[Tuple]] or None (default)
        List of filters to apply, like ``[[('x', '=', 0), ...], ...]``. This
        implements partition-level (hive) filtering only, i.e., to prevent the
        loading of some files of the dataset.

        Predicates are expressed in disjunctive normal form (DNF). This means
        that the innermost tuple describe a single column predicate. These
        inner predicate make are all combined with a conjunction (AND) into a
        larger predicate. The most outer list then combines all filters
        with a disjunction (OR). By this, we should be able to express all
        kinds of filters that are possible using boolean logic.

        This function also supports passing in as List[Tuple]. These predicates
        are evaluated as a conjunction. To express OR in predictates, one must
        use the (preferred) List[List[Tuple]] notation.
    :return: pandas.DataFrame
    """
    original_fallback_path = copy(fallback_path)
    if partition_filters:
        for filter in partition_filters:
            path += '/{}={}'.format(filter[0], filter[1])
            if fallback_path:
                fallback_path += '/{}={}'.format(filter[0], filter[1])

    if not fs and not os.path.exists(path):
        assert fallback_path, 'local execution is turned on and {} does not exists on machine, but fallback_path has not been set'
        print('{} does not exist on local machine, so it\'ll be copied from s3'.format(path))
        ensure_path(path, del_if_exists=False, include_last=True)
        s3 = s3fs.S3FileSystem(anon=False)
        files = s3.ls(fallback_path, detail=False)

        if len(files) == 0:
            table_exists = len(s3.ls(original_fallback_path, detail=False)) > 0
            if table_exists:
                raise AssertionError('The table {} exist, but there are no data for the selected {} partition filters'
                                     .format(original_fallback_path, partition_filters))
            else:
                raise AssertionError('The table {} does not exist')
        for file in files:
            print('fetching ', file)
            s3.get(file, os.path.join(path, file.split('/')[-1]))

    print('Loading board from ', path, verbosity=1)

    non_partition_filters_columns = [filter[0] for filter in non_partition_filters] if non_partition_filters else []

    df = pq.ParquetDataset(path,
                           filesystem=fs,
                           filters=non_partition_filters)\
        .read_pandas(columns=list(set(columns + non_partition_filters_columns)) if columns else None).to_pandas()  # WARNING the filters argument is interfaced as of now but not actually implemented
    if non_partition_filters:
        for column, evaluator, value in non_partition_filters:
            if evaluator == '=':
                df = df[df[column] == value]
            else:
                raise NotImplementedError('{} filter condition not implemented. I suggest you implement it now. It\'s not that hard'.format(evaluator))
    if columns:
        df = df[columns]
    assert df.shape[0] > 0, 'The table {} exist, and there are data for the selected partition filters, but there are no data for the selected {} non-partition filters'\
        .format(original_fallback_path, non_partition_filters)
    return df
