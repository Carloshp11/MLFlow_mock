from datetime import datetime as dt

import pandas as pd
import numpy as np
import s3fs
from dateutil.relativedelta import relativedelta
from sklearn.metrics import confusion_matrix

from library.code_patterns import AttDict
from library.debug import conditional_print as print
from library.io import write_to_s3_parquet, read_parquet
from support_modules.arg_validators import *
from support_modules.config import ChurnConfig

parser = argparse.ArgumentParser()
parser.add_argument('-brand', '--brand', required=True, type=valid_brand)
parser.add_argument("-pd", "--predict_dates",
                    help="intdates argument must be a formatted as [YYYYMMDD, YYYYMMDD, ...] where there can be as many YYYYMMDD "
                         "formatted dates as required. In case only one is inputted, the wraping [] can be ommited.",
                    required=True, type=valid_intdates)
parser.add_argument('-months', '--months_ahead', required=True,
                    help='Number of months to validate the predictions upon. Can be a integer or a string formatted as: '
                         '"[1,2,3]" to implement multiple validation horizons',
                    type=valid_validation_horizons)
parser.add_argument('-env', '--env', required=True, type=valid_env)
parser.add_argument('-mversion', '--model_version', required=True, type=int)
parser.add_argument('-v', '--verbosity', required=True, type=int)

pyargs = parser.parse_args()
print(set_execution_verbosity=pyargs.verbosity)

conf = ChurnConfig(config_path='conf.yaml')
conf.post_process_dinamic_params(pyargs)
conf.config_compliance_checks()
cnames = AttDict(conf.colnames)

s3 = s3fs.S3FileSystem(anon=False) if not conf.local_execution else None
keys = ['customerid', 'countrycode', 'subscriptionid', 'subscriberid']


def assert_cols(df_: pd.DataFrame, cols_to_assert: tuple):
    assert len(set(cols_to_assert).intersection(set(df_.columns))) == len(cols_to_assert)


def alternative_get_confusion_matrix_for_threshold(df_: pd.DataFrame, threshold: float, label_col='label',
                                                   proba_col='proba') -> pd.DataFrame:
    assert isinstance(df_, pd.DataFrame)
    assert_cols(df_, (label_col, proba_col))
    predict_col = df_[proba_col] >= threshold
    label_bools = df_[label_col].astype(bool)

    tp = np.sum(predict_col & label_bools)
    tn = np.sum(~predict_col & ~label_bools)
    fp = np.sum(predict_col & ~label_bools)
    fn = np.sum(~predict_col & label_bools)

    return pd.DataFrame({'threshold': threshold, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}, index=[threshold])


def get_confusion_matrix_for_threshold(df_: pd.DataFrame, threshold: float, label_col='label',
                                       proba_col='proba') -> pd.DataFrame:
    assert isinstance(df_, pd.DataFrame)
    assert_cols(df_, (label_col, proba_col))
    predict_col = (df_[proba_col] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(df_[label_col], predict_col).ravel()
    yield pd.DataFrame({'threshold': threshold, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}, index=[threshold])


def obtain_metrics(df_: pd.DataFrame, n_predictions: int) -> pd.DataFrame:
    df_['accuracy'] = (df_['tp'] + df_['tn']) / n_predictions
    df_['precision'] = df_['tp'] / (df_['tp'] + df_['fp'])
    df_['recall'] = df_['tp'] / (df_['tp'] + df_['fn'])
    df_['pr_A'] = (df_['tp'] + df_['fn']) / n_predictions
    df_['pr_B'] = (df_['tp'] + df_[
        'fp']) / n_predictions  # TODO Miguel, ¿estamos seguros que es (tp + fp)/all y no (tn + fp)/all?
    df_['pond_true'] = df_['pr_A'] * df_['pr_B']
    df_['pond_false'] = (1.0 - df_['pr_A']) * (1.0 - df_['pr_B'])
    df_['pr_e'] = df_['pond_true'] - df_['pond_false']
    df_['kappa'] = (df_['accuracy'] - df_['pr_e']) / (1.0 - df_['pr_e'])
    df_['f1_score'] = 2.0 * (df_['recall'] * df_['precision']) / (df_['recall'] + df_['precision'])
    df_ = df_.drop(['pr_A', 'pr_B', 'pr_e', 'pond_true', 'pond_false'], axis=1)
    df_[['accuracy', 'precision', 'recall', 'kappa', 'f1_score']] = df_[
        ['accuracy', 'precision', 'recall', 'kappa', 'f1_score']].fillna(0)
    return df_


def compute_validation(predicted_date: str, months_ahead: List[int]):
    validation_dates = [(dt.strptime(predicted_date, '%Y%m%d') + relativedelta(months=i)).strftime('%Y%m%d')
                        for i in months_ahead]
    print("Predictions for {} will be validated against data on {}".format(predicted_date, validation_dates),
          verbosity=1)

    # TODO quitar modelversion
    # noinspection PyTypeChecker
    predict_df = read_parquet(s3, path=conf.predict_path,
                              fallback_path=conf.predict_path_fallback,
                              columns=keys + ['modelversion', 'distance'],
                              partition_filters=[(cnames.BRANDID, conf.brand_id),
                                                 (cnames.DATE, predicted_date)],
                              non_partition_filters=[(cnames.MODELVERSION, '=', pyargs.model_version)])
    assert len(list(set(list(predict_df['modelversion'])))) == 1, str(
        set(list(predict_df['modelversion'])))  # TODO quitar
    assert list(set(list(predict_df['modelversion'])))[0] == pyargs.model_version, str(
        set(list(predict_df['modelversion'])))  # TODO quitar
    print('predict_df loaded', verbosity=1)
    predict_df = predict_df.rename(columns={'distance': 'proba'})
    n_predictions = predict_df.shape[0]

    for validation_date in validation_dates:
        # noinspection PyTypeChecker
        reguser_df = read_parquet(s3, path=conf.registered_user_path,
                                  fallback_path=conf.registered_user_path_fallback,
                                  columns=keys,
                                  partition_filters=[(cnames.BRANDID, conf.brand_id),
                                                     (cnames.DAYDATE, validation_date)],
                                  non_partition_filters=[('status', '=', 'PayingSubscription'),
                                                         ('producttype', '=', 'Standalone'),
                                                         ('deactivationdaydate', '=', '20991231')])
        print('reguser_df loaded', verbosity=1)

        reguser_df['active'] = True
        validation_df = predict_df.merge(reguser_df, on=keys, how='left')
        validation_df['active'] = validation_df['active'].fillna(False)
        validation_df['label'] = ~validation_df['active']
        print('Validation dataset created', verbosity=1)
        del reguser_df

        conf_matrix = pd.concat([alternative_get_confusion_matrix_for_threshold(validation_df, threshold=i / 100)
                                 for i in range(100)])
        print('Confusion matrix calculated'.format(validation_date), verbosity=1)
        del validation_df

        metrics = obtain_metrics(conf_matrix, n_predictions=n_predictions)
        print('Metrics for validation date {} calculated'.format(validation_date), verbosity=1)
        del conf_matrix

        metrics[cnames.MODELVERSION] = pyargs.model_version
        metrics[cnames.DATE] = predicted_date
        metrics[cnames.VALIDATIONDATE] = validation_date
        metrics[cnames.BRANDID] = conf.brand_id
        print('Storing on S3 metrics from {0} validated against {1}'.format(predicted_date, validation_date))
        write_to_s3_parquet(s3, df=metrics,
                            path=conf.metrics_path, partition_cols=[cnames.BRANDID, cnames.DATE])
        del metrics
    return None

for predict_date in pyargs.predict_dates:
    compute_validation(predict_date, pyargs.months_ahead)
print('Execution finised')
