import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
import lightgbm as lgb
import torch
from torch import nn
from torch.utils import data
from fastai.train import DataBunch
from fastai.train import Learner

from model_tools import train_model_per_epoch
from metrics import calculate_overall_auc, compute_bias_metrics_for_model, get_final_metric


TRAIN_PREDICTIONS_PATH = [
    # 'predictions/bilstm_ftglove_smartprocess_128x512x256x128_dr03_6folds/cv_train.csv',  # 0.93830
    # 'predictions/bilstm_fttwitter_smartprocess_128x512x256x128_dr03_6folds/cv_train.csv',  # 0.9366
    'predictions/bert_cv/bert_cv_train.csv',
    'predictions/bilstm_ftglove_smartprocess_128x512x256x128_dr03_5folds/cv_train.csv',
    'predictions/lstmgrucnn_ftglove_smartprocess_128x512x512x128_dr03_5folds/cv_lstmgrucnn_train.csv',
    'predictions/cnn_ftglove_smartprocess_128_dr03_5folds/cv_train.csv'
]

CV_PREDICTIONS_PATH = [
    'predictions/bert_cv/bert_cv_submissions.csv',
    'predictions/bilstm_ftglove_smartprocess_128x512x256x128_dr03_5folds/cv_submission_5folds.csv',
    'predictions/lstmgrucnn_ftglove_smartprocess_128x512x512x128_dr03_5folds/cv_lstmgrucnn_submission.csv',
    'predictions/cnn_ftglove_smartprocess_128_dr03_5folds/cv_submission_cnn.csv'
]

LGB_PARAMS = {
    'objective': 'binary',
    'boost_from_average': False,
    'metric': 'auc',
    'max_depth': 15,
    'num_leaves': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.6,
    'bagging_freq': 1,
    'learning_rate': 0.02,
    'num_threads': 6,
    'verbose': -1
}
N_ESTIMATORS = 2000
N_SPLITS = 10
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

np.random.seed(18)

print("Loading cv predictions ...", end="\r", flush=True)
train = pd.read_csv('datas/train.csv')
x_df = pd.read_csv(TRAIN_PREDICTIONS_PATH[0], usecols=['prediction'])
x_df.rename(columns={'prediction': 'prediction_0'}, inplace=True)
for i, path in enumerate(TRAIN_PREDICTIONS_PATH[1:]):
    predictions = pd.read_csv(path, usecols=['prediction'])
    predictions.rename(columns={'prediction': 'prediction' + str(i + 1)}, inplace=True)
    x_df = pd.concat([x_df, predictions], axis=1)

test = pd.read_csv('datas/test.csv')
x_test_df = pd.read_csv(CV_PREDICTIONS_PATH[0], usecols=['prediction'])
x_test_df.rename(columns={'prediction': 'prediction_0'}, inplace=True)
for i, path in enumerate(CV_PREDICTIONS_PATH[1:]):
    cv_predictions = pd.read_csv(path, usecols=['prediction'])
    cv_predictions.rename(columns={'prediction': 'prediction' + str(i + 1)}, inplace=True)
    x_test_df = pd.concat([x_test_df, cv_predictions], axis=1)
print("Loading cv predictions ==> done")

x = x_df.values
y_df = pd.read_csv('datas/train.csv', usecols=['target'])
y = (y_df['target'].values > 0.5).astype(int)
for col in ['target'] + IDENTITY_COLUMNS:
    train[col] = np.where(train[col] >= 0.5, True, False)
x_test = x_test_df.values
y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values
# Overall
weights = np.ones((len(x),)) / 4
# Subgroup
weights += (train[IDENTITY_COLUMNS].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
# Background Positive, Subgroup Negative
weights += (((train['target'].values >= 0.5).astype(bool).astype(np.int) +
             (train[IDENTITY_COLUMNS].fillna(0).values < 0.5).sum(
                 axis=1).astype(bool).astype(np.int)) > 1).astype(bool).astype(np.int) / 4
# Background Negative, Subgroup Positive
weights += (((train['target'].values < 0.5).astype(bool).astype(np.int) +
             (train[IDENTITY_COLUMNS].fillna(0).values >= 0.5).sum(
                 axis=1).astype(bool).astype(np.int)) > 1).astype(bool).astype(np.int) / 4
loss_weight = 1.0 / weights.mean()
y = np.vstack([(train['target'].values >= 0.5).astype(np.int), weights]).T
# y_val = np.vstack([(val['toxic'].values >= 0.5).astype(np.int), np.ones((len(x_val),)) / 4.0]).T


def custom_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:, 1:2])(data[:, :1], targets[:, :1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:], targets[:, 2:])
    return (bce_loss_1 * loss_weight) + bce_loss_2


class MLP(nn.Module):
    def __init__(self, num_aux_targets):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_aux_targets + 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


kf = KFold(n_splits=N_SPLITS)
y_train_predictions = np.zeros(len(train))
test_predictions = np.zeros(len(test))
cv_scores = []
print("Starting stacking cross validation")
for i, (train_index, val_index) in enumerate(kf.split(x)):

    train_x, val_x = x[train_index], x[val_index]
    train_y, val_y = y[train_index], y[val_index]
    x_train_torch = torch.tensor(train_x, dtype=torch.float32)
    x_val_torch = torch.tensor(val_x, dtype=torch.float32)
    y_train_torch = torch.tensor(train_y, dtype=torch.float32)
    y_val_torch = torch.tensor(val_y, dtype=torch.float32)


    cv_train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
    cv_fake_val_dataset = data.TensorDataset(x_train_torch[: 1000], y_train_torch[: 1000])
    cv_val_dataset = data.TensorDataset(x_val_torch, y_val_torch)

    cv_train_loader = data.DataLoader(cv_train_dataset, batch_size=512, shuffle=True)
    cv_fake_val_loader = data.DataLoader(cv_fake_val_dataset, batch_size=512, shuffle=False)
    cv_databunch = DataBunch(train_dl=cv_train_loader, valid_dl=cv_fake_val_loader)

    cv_model = MLP(y_aux_train.shape[-1])
    cv_learn = Learner(cv_databunch, cv_model, loss_func=custom_loss)

    cv_predictions, _, _, _, _ = train_model_per_epoch(cv_learn, cv_val_dataset, output_dim=7, model_idx=0, lr=0.001,
                                                       lr_decay=1, n_epochs=10, save_models='last',
                                                       model_name='mlp_stacking')

    # cv_test_predictions = cv_model.predict(x_test)
    y_train_predictions[val_index] = cv_predictions
    # test_predictions += cv_test_predictions / float(N_SPLITS)

    val_df = train.iloc[val_index].copy()
    val_df['model'] = cv_predictions
    bias_metrics_df = compute_bias_metrics_for_model(val_df, IDENTITY_COLUMNS, 'model', 'target')
    cv_score = get_final_metric(bias_metrics_df, calculate_overall_auc(val_df, 'model'))
    cv_scores.append(cv_score)
    print("Split: {}, cv_score: {:4.5}".format(i, cv_score))

print("All scores: ", cv_scores)
print("Final_score: {:4.5f}".format(np.mean(cv_scores)))

cv_test_preds_df = pd.DataFrame.from_dict({
    'id': test['id'],
    'prediction': test_predictions
})
cv_test_preds_df.to_csv('predictions/stacking_cv_submission.csv', index=False)
