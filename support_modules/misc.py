import pickle

import boto3
import pandas as pd
import s3fs


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