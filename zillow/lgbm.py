def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import lightgbm as lgb
from tools import make_all_predictions, make_submission, make_advanced_predictions
from feature_eng import train_feature_engineering
from model_params import lgb_params

PREDICT_AND_SUBMIT = True

print("preparing datas", end="...", flush=True)
df_train = pd.read_csv("datas/train.csv")
upper_limit = 0.4  # np.percentile(df_train["logerror"].values, 99)
lower_limit = -0.4  # np.percentile(df_train["logerror"].values, 1)
df_train = df_train[df_train["logerror"] < upper_limit]
df_train = df_train[df_train["logerror"] > lower_limit]

x = train_feature_engineering(df_train)
train_columns = x.columns
y = df_train['logerror'].values

split = 45000
train_x, train_y, val_x, val_y = x[:split], y[:split], x[split:], y[split:]

d_train = lgb.Dataset(train_x, label=train_y)
d_valid = lgb.Dataset(val_x, label=val_y)

watchlist = [d_valid]
print(" done", flush=True)
print("\n")

print("training...")
params = lgb_params
clf = lgb.train(params, d_train, 500, watchlist, early_stopping_rounds=100, verbose_eval=100)
print("training done")
print("\n")

if PREDICT_AND_SUBMIT:
    test_filenames = ["test_part1.csv", "test_part2.csv", "test_part3.csv", "test_part4.csv"]

    d_train = lgb.Dataset(x, label=y)
    clf = lgb.train(params, d_train, 500)
    make_advanced_predictions(clf, train_columns, test_filenames, "test.csv")
    """
    predictions = make_all_predictions(clf, train_columns,
                                       ["test_part1.csv", "test_part2.csv", "test_part3.csv", "test_part4.csv"])
    make_submission(predictions, "lgb_benchmark.csv")
    """
