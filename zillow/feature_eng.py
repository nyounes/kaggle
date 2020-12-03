import numpy as np
import pandas as pd


def remove_non_features_col(train_df):
    train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate',
                              'propertyzoningdesc', 'propertycountylandusecode'],
                              axis=1)
    return train_df


def handle_nan(df_train):
    df_train["poolcnt"][np.isnan(df_train["poolcnt"])] = 0
    return df_train


def remove_outliers(df_train):
    upper_limit = np.percentile(df_train["logerror"].values, 99)
    lower_limit = np.percentile(df_train["logerror"].values, 1)
    df_train['logerror'].ix[df_train['logerror'] > upper_limit] = upper_limit
    df_train['logerror'].ix[df_train['logerror'] < lower_limit] = lower_limit
    # df_train = df_train[df_train["logerror"] < upper_limit]
    # df_train = df_train[df_train["logerror"] > lower_limit]
    return df_train


def handle_date(df_train):
    transaction_date = pd.to_datetime(df_train["transactiondate"])
    df_train["transaction_year"] = transaction_date.dt.year
    df_train["transaction_month"] = transaction_date.dt.month
    return df_train


def train_feature_engineering(df_train):
    # df_train = handle_date(df_train)

    df_train = remove_outliers(df_train)
    df_train = remove_non_features_col(df_train)

    for c in df_train.dtypes[df_train.dtypes == object].index.values:
        df_train[c] = (df_train[c] == True)
    return df_train


def test_feature_engineering(df_test, train_columns, date_to_predict):
    """
    year = int(date_to_predict[:4])
    month = int(date_to_predict[4:])
    df_test["transaction_year"] = year
    df_test["transaction_month"] = month
    """

    df_test = df_test[train_columns]
    for c in df_test.dtypes[df_test.dtypes == object].index.values:
        df_test[c] = (df_test[c] == True)
    return df_test
