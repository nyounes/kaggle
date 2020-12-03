import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
import lightgbm as lgb

from metrics import calculate_overall_auc, compute_bias_metrics_for_model, get_final_metric


DATA_PATH = "/media/nyounes/6ed30f7e-b07b-42c4-a244-854b5da6f9a6/kaggle/jigsaw/predictions/"

TRAIN_PREDICTIONS_PATH = [
    # 'predictions/bilstm_ftglove_smartprocess_128x512x256x128_dr03_6folds/cv_train.csv',  # 0.93830
    # 'predictions/bilstm_fttwitter_smartprocess_128x512x256x128_dr03_6folds/cv_train.csv',  # 0.9366
    'bert_cv/bert_cv_train.csv',
    'bilstm_ftglove_smartprocess_128x512x256x128_dr03_5folds/cv_train.csv',
    'lstmgrucnn_ftglove_smartprocess_128x512x512x128_dr03_5folds/cv_lstmgrucnn_train.csv',
    'cnn_ftglove_smartprocess_128_dr03_5folds/cv_train.csv',
    'bilstm_ftglove_lightprocess_128x512x256x128_dr03_5folds/cv_train.csv',
    'lstmgrucnn_ftglove_lightprocess_128x512x512x128_dr03_5folds/cv_train.csv',
    'cnn_ftglove_lightprocess_128_dr03_5folds/cv_train.csv',
]

CV_PREDICTIONS_PATH = [
    'bert_cv/bert_cv_submissions.csv',
    'bilstm_ftglove_smartprocess_128x512x256x128_dr03_5folds/cv_submission_5folds.csv',
    'lstmgrucnn_ftglove_smartprocess_128x512x512x128_dr03_5folds/cv_lstmgrucnn_submission.csv',
    'cnn_ftglove_smartprocess_128_dr03_5folds/cv_submission_cnn.csv',
    'bilstm_ftglove_lightprocess_128x512x256x128_dr03_5folds/cv_submission.csv',
    'lstmgrucnn_ftglove_lightprocess_128x512x512x128_dr03_5folds/cv_submission.csv',
    'cnn_ftglove_lightprocess_128_dr03_5folds/cv_submission.csv',
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

def add_features(df):
    df['comment_text'] = df['comment_text'].apply(lambda x: str(x))
    df['total_length'] = df['comment_text'].apply(len)
    df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']), axis=1)
    df['num_words'] = df.comment_text.str.count('\S+')
    df['num_unique_words'] = df['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    return df

print("Loading cv predictions ...", end="\r", flush=True)
train = pd.read_csv('datas/train.csv')
x_df = pd.read_csv(DATA_PATH + TRAIN_PREDICTIONS_PATH[0], usecols=['prediction'])
x_df.rename(columns={'prediction': 'prediction_0'}, inplace=True)
for i, path in enumerate(TRAIN_PREDICTIONS_PATH[1:]):
    predictions = pd.read_csv(DATA_PATH + path, usecols=['prediction'])
    predictions.rename(columns={'prediction': 'prediction' + str(i + 1)}, inplace=True)
    x_df = pd.concat([x_df, predictions], axis=1)
add_features(train)
train_features = train[['total_length', 'capitals', 'caps_vs_length', 'num_words', 'num_unique_words', 'words_vs_unique']]
x_df = pd.concat([x_df, train_features], axis=1)

test = pd.read_csv('datas/test.csv')
x_test_df = pd.read_csv(DATA_PATH + CV_PREDICTIONS_PATH[0], usecols=['prediction'])
x_test_df.rename(columns={'prediction': 'prediction_0'}, inplace=True)
for i, path in enumerate(CV_PREDICTIONS_PATH[1:]):
    cv_predictions = pd.read_csv(DATA_PATH + path, usecols=['prediction'])
    cv_predictions.rename(columns={'prediction': 'prediction' + str(i + 1)}, inplace=True)
    x_test_df = pd.concat([x_test_df, cv_predictions], axis=1)
add_features(test)
test_features = test[['total_length', 'capitals', 'caps_vs_length', 'num_words', 'num_unique_words', 'words_vs_unique']]
x_test_df = pd.concat([x_test_df, test_features], axis=1)
print("Loading cv predictions ==> done")

x = x_df.values
y_df = pd.read_csv('datas/train.csv', usecols=['target'])
y = (y_df['target'].values > 0.5).astype(int)
for col in ['target'] + IDENTITY_COLUMNS:
    train[col] = np.where(train[col] >= 0.5, True, False)
x_test = x_test_df.values

kf = KFold(n_splits=N_SPLITS)
y_train_predictions = np.zeros(len(train))
test_predictions = np.zeros(len(test))
cv_scores = []
print("Starting stacking cross validation")
for i, (train_index, val_index) in enumerate(kf.split(x)):

    train_x, val_x = x[train_index], x[val_index]
    train_y, val_y = y[train_index], y[val_index]

    dataset_train = lgb.Dataset(train_x, label=train_y,
                                feature_name=list(x_df.columns))
    dataset_val = lgb.Dataset(val_x, label=val_y)

    cv_model = lgb.train(LGB_PARAMS, dataset_train, N_ESTIMATORS, valid_sets=dataset_val, early_stopping_rounds=200,
                         verbose_eval=0)
    cv_predictions = cv_model.predict(val_x)
    cv_test_predictions = cv_model.predict(x_test)
    print(cv_test_predictions[:10])
    y_train_predictions[val_index] = cv_predictions
    test_predictions += cv_test_predictions / float(N_SPLITS)

    val_df = train.iloc[val_index].copy()
    val_df['model'] = cv_predictions
    bias_metrics_df = compute_bias_metrics_for_model(val_df, IDENTITY_COLUMNS, 'model', 'target')
    cv_score = get_final_metric(bias_metrics_df, calculate_overall_auc(val_df, 'model'))
    cv_scores.append(cv_score)

    cv_model.save_model('stacking_cv' + str(i + 1) + '.txt')
    print("Split: {}, cv_score: {:4.5}".format(i, cv_score))
    print('\n')

print("All scores: ", cv_scores)
print("Final_score: {:4.5f}".format(np.mean(cv_scores)))

cv_test_preds_df = pd.DataFrame.from_dict({
    'id': test['id'],
    'prediction': test_predictions
})
cv_test_preds_df.to_csv('predictions/stacking_cv_submission.csv', index=False)
