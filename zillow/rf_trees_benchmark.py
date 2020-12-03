import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
from tools import make_all_predictions, make_submission

df_train = pd.read_csv("datas/train.csv")

x = df_train.drop(['parcelid', 'logerror', 'transactiondate',
                   'propertyzoningdesc', 'propertycountylandusecode'],
                  axis=1)
train_columns = x.columns
y = df_train['logerror'].values

for c in x.dtypes[x.dtypes == object].index.values:
    x[c] = (x[c] == True)

rf = RandomForestRegressor()
mae_scorer = make_scorer(mean_absolute_error)
rf_scores = cross_val_score(rf, x, y, scoring=mae_scorer)
print("min score: {0} -- max score: {1} -- mean score {2}".format(
    np.min(rf_scores), np.max(rf_scores), np.mean(rf_scores)))


predictions = make_all_predictions(clf, ["test_part1.csv", "test_part2.csv"])
print(predictions.shape)
make_submission(predictions)
