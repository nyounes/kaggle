import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import DataLoader

from utils import load_config, seed_everything
from embedding_features import get_embedding_features, get_dist_features
from embedding_tools import build_embedding_matrix
from dataset_tools import GQBaselineDataset
from pytorch_models.baseline_model import GQBaselineModel
from model_tools import train_model, make_prediction
from ftglove_preprocessing import preprocess


config = load_config('config.yaml')

EMBEDDING_INPUTS = ['crawl-300d-2M']
max_len_body = config['max_len_body']
max_len_title = config['max_len_title']
max_len_answer = config['max_len_answer']
batch_size = config['batch_size']
epochs = config['epochs']
n_folds = config['n_folds']
lr = config['lr']
log_folder = config['log_folder']

sample_submission = pd.read_csv('datas/sample_submission.csv')
test = pd.read_csv('datas/test.csv', nrows=20).fillna(' ')
train = pd.read_csv('datas/train.csv', nrows=100).fillna(' ')
print("shape of train data: ", train.shape)
print("shape of test data: ", test.shape)

train['answer'] = train['answer'].apply(lambda x: preprocess(x))
train['question_body'] = train['question_body'].apply(lambda x: preprocess(x))
train['question_title'] = train['question_title'].apply(lambda x: preprocess(x))
test['answer'] = test['answer'].apply(lambda x: preprocess(x))
test['question_body'] = test['question_body'].apply(lambda x: preprocess(x))
test['question_title'] = test['question_title'].apply(lambda x: preprocess(x))

seed_everything()

embedding_train, embedding_test = get_embedding_features(config['use_model_path'], train, test,
                                                         ['answer', 'question_body', 'question_title'])

dist_features_train, dist_features_test = get_dist_features(embedding_train, embedding_test)

full_text = list(train['question_body']) + \
                       list(train['answer']) + \
                       list(train['question_title']) + \
                       list(test['question_body']) + \
                       list(test['answer']) + \
                       list(test['question_title'])

tokenizer = Tokenizer(filters='', lower=False)
tokenizer.fit_on_texts(full_text)

embedding_matrix = build_embedding_matrix(EMBEDDING_INPUTS, tokenizer, config)

unique_hosts = list(set(train['host'].unique().tolist() + test['host'].unique().tolist()))
host_dict = {i + 1: e for i, e in enumerate(unique_hosts)}

unique_categories = list(set(train['category'].unique().tolist() + test['category'].unique().tolist()))
category_dict = {i + 1: e for i, e in enumerate(unique_categories)}

train_question_tokenized = pad_sequences(tokenizer.texts_to_sequences(train['question_body']), maxlen=max_len_body)
train_title_tokenized = pad_sequences(tokenizer.texts_to_sequences(train['question_title']), maxlen=max_len_title)
train_answer_tokenized = pad_sequences(tokenizer.texts_to_sequences(train['answer']), maxlen=max_len_answer)

test_question_tokenized = pad_sequences(tokenizer.texts_to_sequences(test['question_body']), maxlen=max_len_body)
test_title_tokenized = pad_sequences(tokenizer.texts_to_sequences(test['question_title']), maxlen=max_len_title)
test_answer_tokenized = pad_sequences(tokenizer.texts_to_sequences(test['answer']), maxlen=max_len_answer)

y = train[sample_submission.columns[1:]].values
test_loader = DataLoader(
    GQBaselineDataset(test_question_tokenized, test_answer_tokenized, test_title_tokenized,
                      embedding_test, dist_features_test, test.index),
    batch_size=batch_size, shuffle=False, num_workers=0
)

folds = KFold(n_splits=n_folds)
preds = np.zeros((len(test), epochs))
best_scores = []
best_epochs = []
best_train_losses = []
best_val_losses = []

for fold_n, (train_index, valid_index) in enumerate(folds.split(train)):
    print(f'Fold {fold_n + 1} started at {time.ctime()}')
    train_loader = DataLoader(
        GQBaselineDataset(train_question_tokenized, train_answer_tokenized, train_title_tokenized,
                          embedding_train, dist_features_train, train_index, y),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    valid_loader = DataLoader(
        GQBaselineDataset(train_question_tokenized, train_answer_tokenized, train_title_tokenized,
                          embedding_train, dist_features_train, valid_index, y),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    model = GQBaselineModel(embedding_matrix=embedding_matrix)
    model.cuda()

    model, score, epoch, train_loss, val_loss = train_model(model, train_loader, valid_loader, n_epochs=epochs, lr=lr)
    best_scores.append(score)
    best_epochs.append(epoch)
    best_train_losses.append(train_loss)
    best_val_losses.append(val_loss)
    torch.save(model.state_dict(), log_folder + 'pytorch_baseline' + str(fold_n) + '.bin')
    prediction = make_prediction(test_loader, model)
    preds += prediction / folds.n_splits

    print()

print()
for i in range(n_folds):
    print('Epoch {}/{} \t best_loss={:.4f} \t best_val_loss={:.4f} \t best_score={:.4f}'.format(
        i, n_folds, best_train_losses[i], best_val_losses[i], best_scores[i]))

print()
print("avg_train_loss={:.4f} \t avg_val_loss={:.4f} \t avg_score={:.4f}".format(
    np.mean(best_train_losses), np.mean(best_val_losses), np.mean(best_scores)))

sample_submission.loc[:, 'question_asker_intent_understanding':] = np.clip(preds, 0.00001, 0.999999)
sample_submission.to_csv('submission.csv', index=False)
