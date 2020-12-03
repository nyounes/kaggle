import pandas as pd
import xgboost as xgb
from tools import make_all_predictions, make_submission

df_train = pd.read_csv("datas/train.csv")

x = df_train.drop(['parcelid', 'logerror', 'transactiondate',
                   'propertyzoningdesc', 'propertycountylandusecode'],
                  axis=1)
# train_columns = ["taxamount", "structuretaxvaluedollarcnt", "landtaxvaluedollarcnt"]
# x = x[train_columns]
train_columns = x.columns
y = df_train['logerror'].values

for c in x.dtypes[x.dtypes == object].index.values:
    x[c] = (x[c] == True)

split = 50000
train_x, train_y, val_x, val_y = x[:split], y[:split], x[split:], y[split:]

print("building DMatrix...")

d_train = xgb.DMatrix(train_x, label=train_y)
d_valid = xgb.DMatrix(val_x, label=val_y)

params = {"eta": 0.02, "objective": "reg:linear", "eval_metric": "mae", "max_depth": 2, "silent": 1}

watchlist = [(d_train, 'train'), (d_valid, 'validation')]

clf = xgb.train(params, d_train, 10000, watchlist,
                early_stopping_rounds=100, verbose_eval=10)

predictions = make_all_predictions(clf, ["test_part1.csv", "test_part2.csv"])
print(predictions.shape)
make_submission(predictions)
