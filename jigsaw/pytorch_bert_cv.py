import warnings
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch import nn
import torch.utils.data
from apex import amp

from logger import logger
from utils import load_config, seed_everything
from loss_weight import calculate_loss_weight
from bert_formatter import convert_lines
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import BertConfig
from metrics import calculate_overall_auc, compute_bias_metrics_for_model, get_final_metric

warnings.filterwarnings(action='once')


def custom_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:, 1:2])(data[:, :1], targets[:, :1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:], targets[:, 2:])
    return (bce_loss_1 * loss_weight) + bce_loss_2


parser = argparse.ArgumentParser()
parser.add_argument('--split', default='cv')
args = parser.parse_args()
current_split = int(args.split)

config = load_config('config.yaml')
device = torch.device('cuda')
max_len = config['max_len']
bert_model = config['bert']['model']
lr = config['bert']['lr']
batch_size = config['bert']['batch_size']
accumulation_steps = config['bert']['accumulation_steps']
epochs = config['epochs']
output_model_path = config['bert']['output_model_path']
bert_config = BertConfig(config['bert']['config_path'])
n_folds = config['n_folds']
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

seed_everything()
logger.info("Loading pre trained BERT tokenizer ...")
tokenizer = BertTokenizer.from_pretrained(bert_model, cache_dir=None, do_lower_case=True)
logger.info("Loading pre trained BERT tokenizer ==> done")

logger.info("Creating train and test datasets ...")
train_df = pd.read_csv("datas/train.csv", nrows=10000)
test_df = pd.read_csv("datas/test.csv", nrows=10000)
train_df['comment_text'] = train_df['comment_text'].astype(str)
sequences = convert_lines(train_df["comment_text"].fillna("DUMMY_VALUE"), max_len, tokenizer)
train_df = train_df.fillna(0)
y_columns = ['target']
train_df = train_df.drop(['comment_text'], axis=1)
train_df['target'] = (train_df['target'] >= 0.5).astype(float)

weights, loss_weight = calculate_loss_weight(train_df, IDENTITY_COLUMNS)
y = np.vstack([(train_df['target'].values >= 0.5).astype(np.int), weights]).T
y_aux = train_df[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values

for col in ['target'] + IDENTITY_COLUMNS:
    train_df[col] = np.where(train_df[col] >= 0.5, True, False)

kf = KFold(n_splits=n_folds, shuffle=True)
index_list = list(kf.split(train_df.values))
train_index = index_list[current_split][0]
logger.info(len(train_index))
val_index = index_list[current_split][1]
logger.info(len(val_index))

x = sequences[train_index]
y_train = y[train_index]
y_aux_train = y_aux[train_index]
x_val = sequences[val_index]
y_val = y[val_index]
y_aux_val = y_aux[val_index]
val_df = train_df.iloc[val_index].copy()

x_train_torch = torch.tensor(x, dtype=torch.long)
y_train_torch = torch.tensor(np.hstack([y_train, y_aux_train]), dtype=torch.float32)
train_dataset = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)
logger.info("Creating train and test datasets ==> done")


logger.info("Loading pre trained BERT model ...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", cache_dir=None, num_labels=7)
logger.info("Loading pre trained BERT model ==> done")
model.zero_grad()
model = model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

train = train_dataset

num_train_optimization_steps = int(epochs*len(train)/batch_size/accumulation_steps)

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=lr,
                     warmup=0.05,
                     t_total=num_train_optimization_steps)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
model = model.train()

tq = tqdm(range(epochs))
for epoch in tq:
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    avg_loss = 0.
    avg_accuracy = 0.
    lossf = None
    tk0 = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, (x_batch, y_batch) in tk0:
        optimizer.zero_grad()
        y_pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
        loss = custom_loss(y_pred, y_batch.to(device))
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
        tk0.set_postfix(loss=lossf)
        avg_loss += loss.item() / len(train_loader)
        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:, 0]) > 0.5) == (y_batch[:, 0] > 0.5).to(device)).to(torch.float)).item() / len(train_loader)
    tq.set_postfix(avg_loss=avg_loss, avg_accuracy=avg_accuracy)


torch.save(model.state_dict(), output_model_path + str(current_split + 1) + '.bin')
# model.load_state_dict(torch.load('bert/cv1/bert_pytorch_1.bin'))

for param in model.parameters():
    param.requires_grad = False
model.eval()

y_train_predictions = np.zeros(len(train_df))
x_val_torch = torch.tensor(x_val, dtype=torch.long)
y_val_torch = torch.tensor(np.hstack([y_val, y_aux_val]), dtype=torch.float32)
val_dataset = torch.utils.data.TensorDataset(x_val_torch)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
tk0 = tqdm(val_loader)
val_predictions = np.zeros((len(x_val)))
for i, (x_batch,) in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
    val_predictions[i * 32:(i + 1) * 32] = pred[:, 0].detach().cpu().squeeze().numpy()
val_predictions = torch.sigmoid(torch.tensor(val_predictions)).numpy().ravel()
val_df['model'] = val_predictions
bias_metrics_df = compute_bias_metrics_for_model(val_df, IDENTITY_COLUMNS, 'model', 'target')
cv_score = get_final_metric(bias_metrics_df, calculate_overall_auc(val_df, 'model'))
y_train_predictions[val_index] = val_predictions
logger.info('cv_score for split {}: {}'.format(current_split, cv_score))

train_predictions_df = pd.DataFrame.from_dict({
    'id': train_df['id'],
    'prediction': y_train_predictions
})
train_predictions_df.to_csv('bert/cv_bert_train_predictions_' + str(current_split + 1) + '.csv', index=False)

test_df['comment_text'] = test_df['comment_text'].astype(str)
x_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), max_len, tokenizer)

for param in model.parameters():
    param.requires_grad = False
model.eval()

test_preds = np.zeros((len(x_test)))
test = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.long))
test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
tk0 = tqdm(test_loader)
for i, (x_batch,) in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
    test_preds[i * 32:(i + 1) * 32] = pred[:, 0].detach().cpu().squeeze().numpy()

test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()

submission = pd.DataFrame.from_dict({
    'id': test_df['id'],
    'prediction': test_pred
})
submission.to_csv('bert/bert_submission_' + str(current_split + 1) + '.csv', index=False)
