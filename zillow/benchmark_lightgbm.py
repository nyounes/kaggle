import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from tools import make_all_predictions, make_submission
from feature_eng import feature_engineering

df_train = pd.read_csv("datas/train.csv")

x = feature_engineering(df_train)
train_columns = x.columns
y = df_train['logerror'].values

for c in x.dtypes[x.dtypes == object].index.values:
    x[c] = (x[c] == True)

split = 45000
train_x, train_y, val_x, val_y = x[:split], y[:split], x[split:], y[split:]

print("building DMatrix...")

d_train = lgb.Dataset(train_x, label=train_y)
d_valid = lgb.Dataset(val_x, label=val_y)

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.00233 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'          # or 'mae'
params['sub_feature'] = 0.5      # feature_fraction
params['bagging_fraction'] = 0.7 # sub_row
params['bagging_freq'] = 20
params['num_leaves'] = 60        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = -1

watchlist = [d_valid]

clf = lgb.train(params, d_train, 500, watchlist, early_stopping_rounds=100, verbose_eval=10)

"""
d_train = lgb.Dataset(x, label=y)
clf = lgb.train(params, d_train, 500)

predictions = make_all_predictions(clf, ["test_part1.csv", "test_part2.csv"])
print(predictions.shape)
make_submission(predictions)
"""
