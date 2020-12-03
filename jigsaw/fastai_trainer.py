import os
import gc
from time import time
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from keras.preprocessing import text, sequence
from torch import nn
from fastai.train import Learner

from logger import logger
from utils import load_config, seed_everything
from features import add_features
from loss_weight import calculate_loss_weight

from preprocessing_light_heavy import light_preprocessing
from embedding_tools import build_embedding_matrix
from dataset_tools import create_full_train_datasets, create_cross_val_datasets
from pytorch_models.bi_lstm import BiLSTM
from pytorch_models.lstm_gru_cnn import LstmGruCnn
from pytorch_models.cnn import Cnn
from model_tools import train_model_per_epoch, predict
from metrics import calculate_overall_auc, compute_bias_metrics_for_model, get_final_metric


def custom_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:, 1:2])(data[:, :1], targets[:, :1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:], targets[:, 2:])
    return (bce_loss_1 * loss_weight) + bce_loss_2


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='cv')
args = parser.parse_args()

config = load_config('config.yaml')

EMBEDDING_INPUTS = ['crawl-300d-2M', 'glove.840B.300d']

IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

num_models = config['n_models']
max_len = config['max_len']
batch_size = config['batch_size']
epochs = config['epochs']
n_folds = config['n_folds']
log_folder = config['log_folder']
if not os.path.exists(log_folder):
    os.mkdir(log_folder)

logger.info('script mode: {}'.format(args.mode))
logger.info('num_models: {}'.format(num_models))
logger.info('n_epochs: {}'.format(epochs))
logger.info('max_len: {}'.format(max_len))

seed_everything()

logger.info("Loading train and test datas ...")
start_time = time()
train = pd.read_csv('datas/train_sample.csv')
test = pd.read_csv('datas/test_sample.csv')
train.dropna(subset=['comment_text'], inplace=True)
train = add_features(train)
test = add_features(test)

x_train = eval(config['preprocessing'])(train['comment_text'])
x_test = eval(config['preprocessing'])(test['comment_text'])

features = train[['caps_vs_length', 'words_vs_unique']].fillna(0)
test_features = test[['caps_vs_length', 'words_vs_unique']].fillna(0)
ss = StandardScaler()
ss.fit(np.vstack((features, test_features)))
train_features = ss.transform(features)
test_features = ss.transform(test_features)
y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values
logger.info("Loading train and test datas ==> done in {:4.2f}".format(time() - start_time))

logger.info("Creating target with identity columns ...")
start_time = time()
weights, loss_weight = calculate_loss_weight(train, IDENTITY_COLUMNS)
y_train = np.vstack([(train['target'].values >= 0.5).astype(np.int), weights]).T
logger.info("Creating target with identity columns ==> done in {:4.2f}".format(time() - start_time))

max_features = None
tokenizer = text.Tokenizer(filters='', lower=False)
logger.info("Fitting tokenizer ...")
start_time = time()
tokenizer.fit_on_texts(list(x_train) + list(x_test))
logger.info("Fitting tokenizer ==> done in {:4.2f}".format(time() - start_time))

logger.info("Applying tokenizer and padding ...")
start_time = time()
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
max_features = max_features or len(tokenizer.word_index) + 1
logger.info("Applying tokenizer and padding ==> done in {:4.2f}".format(time() - start_time))

logger.info("Creating embedding matrix and embedding layer...")
start_time = time()
embedding_matrix = build_embedding_matrix(EMBEDDING_INPUTS, tokenizer, config)
gc.collect()
logger.info("Creating embedding matrix and embedding_layer==> done in {:4.2f}".format(time() - start_time))

logger.info('starting script in mode {}'.format(args.mode))
logger.info('model used {}'.format(config['model']))
logger.info('model parameters:')
logger.info(config[config['model']])

if args.mode == 'train':
    final_test_preds = []

    databunch, test_dataset = create_full_train_datasets(x_train, y_train, y_aux_train, train_features,
                                                         x_test, test_features)
    for model_idx in range(num_models):
        logger.info('Model {}'.format(model_idx))
        seed_everything(model_idx)
        model = config['model']
        model = eval(model)(embedding_matrix, max_features, y_aux_train.shape[-1], **config[model])
        model.cuda()
        learn = Learner(databunch, model, loss_func=custom_loss)
        all_test_preds = train_model_per_epoch(
            learn, test_dataset, output_dim=7, model_idx=model_idx, lr=0.001, lr_decay=1, n_epochs=epochs,
            model_name=log_folder + '/train_ftglove_smart_preprocess_'
        )
        checkpoint_weights = [1.5 ** epoch for epoch in range(epochs)]
        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
        final_test_preds.append(test_preds)

    submission = pd.DataFrame.from_dict({
            'id': test['id'],
            'prediction': np.mean(final_test_preds, axis=0)[:, 0]
    })
    submission.to_csv('submission.csv', index=False)


if args.mode == 'cv':
    kf = KFold(n_splits=n_folds, shuffle=True)
    scores = []
    y_train_predictions = np.zeros(len(x_train))
    test_predictions = np.zeros(len(test))

    for col in ['target'] + IDENTITY_COLUMNS:
        train[col] = np.where(train[col] >= 0.5, True, False)

    for i, (train_index, val_index) in enumerate(kf.split(x_train)):
        databunch, val_dataset, test_dataset = create_cross_val_datasets(
            x_train, y_train, y_aux_train, train_features, train_index, val_index, batch_size,
            x_test, test_features)
        seed_everything()

        model = config['model']
        model = eval(model)(embedding_matrix, max_features, y_aux_train.shape[-1], **config[model])
        learner = Learner(databunch, model, loss_func=custom_loss)

        predictions = train_model_per_epoch(
            learner, val_dataset, output_dim=7, model_idx=0, lr=0.001, lr_decay=1, n_epochs=epochs,
            save_models='last', model_name=log_folder + '/fastai_cv' + str(i + 1) + '_ftglove_'
        )
        val_df = train.iloc[val_index].copy()
        val_df['model'] = predictions[-1][:, 0]
        bias_metrics_df = compute_bias_metrics_for_model(val_df, IDENTITY_COLUMNS, 'model', 'target')
        score = get_final_metric(bias_metrics_df, calculate_overall_auc(val_df, 'model'))

        logger.info("Split: {}, score: {:4.5}".format(i, score))
        scores.append(score)
        y_train_predictions[val_index] = predictions[-1][:, 0]

        cv_test_predictions = predict(test_dataset, learner, batch_size=256, output_dim=7)[:, 0]
        test_predictions += cv_test_predictions / n_folds

    logger.info("All scores: ", scores)
    logger.info("Final_score: {:4.5f}".format(np.mean(scores)))
    logger.info("Creating cross val submission file ...")

    cv_test_preds_df = pd.DataFrame.from_dict({
        'id': test['id'],
        'prediction': test_predictions
    })
    cv_test_preds_df.to_csv('predictions/cv_submission.csv', index=False)

    cv_train_preds_df = pd.DataFrame.from_dict({
        'id': train['id'],
        'prediction': y_train_predictions
    })
    cv_train_preds_df.to_csv('predictions/cv_train.csv', index=False)
    logger.info("Creating cross val submission file ==> done")
