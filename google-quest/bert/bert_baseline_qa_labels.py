import sys
import os
import gc
import time
from tqdm import tqdm
import math
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import transformers
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification, BertConfig,
    WEIGHTS_NAME, CONFIG_NAME, AdamW, get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from logger import logger
from utils import load_config, seed_everything
from bert_utils import compute_input_arays, compute_output_arrays
from bert_utils import compute_input_arrays_question, compute_input_arrays_answer
from bert_utils import compute_input_arrays_question_wlabels, compute_input_arrays_answer_wlabels
from dataset import QuestDataset
from custom_bert import CustomBert
from model_tools import train_model, val_model, predict_result


config = load_config('bert/config.yaml')

seed_everything()

max_len_question = config['max_len_question']
max_len_answer = config['max_len_question']
n_folds = config['n_folds']
batch_size = config['batch_size']
epochs = config['epochs']
lr = config['lr']
log_folder = config['log_folder']
accumulation_steps = config['accumulation_steps']
bert_model_config = 'bert/' + config['bert_config_path']
bert_config = BertConfig.from_json_file(bert_model_config)
bert_config.num_labels = 30

bert_model = config['bert_model']
do_lower_case = 'uncased' in bert_model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
question_output_model_path = config['question_output_model_path']
answer_output_model_path = config['answer_output_model_path']

tokenizer = BertTokenizer.from_pretrained("bert/bert-base-uncased/bert-base-uncased-vocab.txt")

TARGET_COLS = ['question_asker_intent_understanding', 'question_body_critical',
               'question_conversational', 'question_expect_short_answer',
               'question_fact_seeking', 'question_has_commonly_accepted_answer',
               'question_interestingness_others', 'question_interestingness_self',
               'question_multi_intent', 'question_not_really_a_question',
               'question_opinion_seeking', 'question_type_choice',
               'question_type_compare', 'question_type_consequence',
               'question_type_definition', 'question_type_entity',
               'question_type_instructions', 'question_type_procedure',
               'question_type_reason_explanation', 'question_type_spelling',
               'question_well_written', 'answer_helpful',
               'answer_level_of_information', 'answer_plausible',
               'answer_relevance', 'answer_satisfaction',
               'answer_type_instructions', 'answer_type_procedure',
               'answer_type_reason_explanation', 'answer_well_written']




train = pd.read_csv("datas/train.csv", nrows=100)
test = pd.read_csv("datas/test.csv", nrows=20)

input_categories = list(train.columns[[1, 2, 5]])
test_inputs = compute_input_arays(test, input_categories, tokenizer, max_sequence_length=512)
lengths_test = np.argmax(test_inputs[0] == 0, axis=1)
lengths_test[lengths_test == 0] = test_inputs[0].shape[1]

kf = GroupKFold(n_splits=n_folds)

test_set = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
result = np.zeros((len(test), 30))

y_train = train[TARGET_COLS].values

logger.info(f"For Every Fold, Train {epochs} Epochs")

for fold, (train_index, val_index) in enumerate(kf.split(X=train.question_body, groups=train.question_body)):
    logger.info("Current Fold: {}".format(fold))

    train_df, val_df = train.iloc[train_index], train.iloc[val_index]

    logger.info("Preparing train datasets....")

    inputs_train_question = compute_input_arrays_question_wlabels(train_df, input_categories, TARGET_COLS[:21],
                                                                  tokenizer, max_sequence_length=max_len_question)
    inputs_train_answer = compute_input_arrays_answer_wlabels(train_df, input_categories, TARGET_COLS[21:],
                                                              tokenizer, max_sequence_length=max_len_answer)
    outputs_train_question = compute_output_arrays(train_df, columns=TARGET_COLS[:21])
    outputs_train_question = torch.tensor(outputs_train_question, dtype=torch.float32)

    outputs_train_answer = compute_output_arrays(train_df, columns=TARGET_COLS[21:])
    outputs_train_answer = torch.tensor(outputs_train_answer, dtype=torch.float32)

    lengths_train_question = np.argmax(inputs_train_question[0] == 0, axis=1)
    lengths_train_question[lengths_train_question == 0] = inputs_train_question[0].shape[1]

    lengths_train_answer = np.argmax(inputs_train_answer[0] == 0, axis=1)
    lengths_train_answer[lengths_train_answer == 0] = inputs_train_answer[0].shape[1]

    logger.info("Preparing Valid datasets....")

    inputs_valid_question = compute_input_arrays_question_wlabels(val_df, input_categories, TARGET_COLS[:21],
                                                                  tokenizer, max_sequence_length=max_len_question)
    inputs_valid_answer = compute_input_arrays_answer_wlabels(val_df, input_categories, TARGET_COLS[21:],
                                                              tokenizer, max_sequence_length=max_len_answer)
    outputs_valid_question = compute_output_arrays(val_df, columns=TARGET_COLS[:21])
    outputs_valid_question = torch.tensor(outputs_valid_question, dtype=torch.float32)
    outputs_valid_answer = compute_output_arrays(val_df, columns=TARGET_COLS[21:])
    outputs_valid_answer = torch.tensor(outputs_valid_answer, dtype=torch.float32)
    lengths_valid_question = np.argmax(inputs_valid_question[0] == 0, axis=1)
    lengths_valid_question[lengths_valid_question == 0] = inputs_valid_question[0].shape[1]
    lengths_valid_answer = np.argmax(inputs_valid_answer[0] == 0, axis=1)
    lengths_valid_answer[lengths_valid_answer == 0] = inputs_valid_answer[0].shape[1]

    logger.info("Preparing Dataloaders Datasets....")

    train_set_question = QuestDataset(inputs=inputs_train_question, lengths=lengths_train_question,
                                      labels=outputs_train_question)
    train_loader_question = DataLoader(train_set_question, batch_size=batch_size, shuffle=True)
    train_set_answer = QuestDataset(inputs=inputs_train_answer, lengths=lengths_train_answer,
                                    labels=outputs_train_answer)
    train_loader_answer = DataLoader(train_set_answer, batch_size=batch_size, shuffle=True)

    valid_set_question = QuestDataset(inputs=inputs_valid_question, lengths=lengths_valid_question,
                                      labels=outputs_valid_question)
    valid_loader_question = DataLoader(valid_set_question, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_set_answer = QuestDataset(inputs=inputs_valid_answer, lengths=lengths_valid_answer,
                                    labels=outputs_valid_answer)
    valid_loader_answer = DataLoader(valid_set_answer, batch_size=batch_size, shuffle=False, drop_last=False)

    # model = BertForSequenceClassification.from_pretrained('bert/bert-base-uncased/', config=bert_config);
    model = CustomBert.from_pretrained('bert/bert-base-uncased/', config=bert_config, num_labels=21)
    model.zero_grad()
    model.to(device)
    torch.cuda.empty_cache()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3, 4, 5, 6], gamma=0.4)
    criterion = nn.BCEWithLogitsLoss()

    logger.info("Training model for questions ...")

    for epoch in tqdm(range(epochs)):

        torch.cuda.empty_cache()

        start_time = time.time()
        avg_loss = train_model(model, train_loader_question, optimizer, criterion)
        avg_val_loss, score = val_model(model, criterion, valid_loader_question, val_shape=val_df.shape[0],
                                        num_labels=21)
        elapsed_time = time.time() - start_time
        scheduler.step()

        logger.info('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t score={:.6f} \t time={:.2f}s'.format(
            epoch + 1, epochs, avg_loss, avg_val_loss, score, elapsed_time))
        torch.save(model.state_dict(),
                   log_folder + question_output_model_path.format(str(fold+1), str(epoch+1)))

    model = CustomBert.from_pretrained('bert/bert-base-uncased/', config=bert_config, num_labels=9)
    model.zero_grad()
    model.to(device)
    torch.cuda.empty_cache()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3, 4, 5, 6], gamma=0.4)
    criterion = nn.BCEWithLogitsLoss()

    logger.info("Training model for answers ...")

    for epoch in tqdm(range(epochs)):

        start_time   = time.time()
        avg_loss     = train_model(model, train_loader_answer, optimizer, criterion)#, scheduler)
        avg_val_loss, score = val_model(model, criterion, valid_loader_answer, val_shape=val_df.shape[0], num_labels=9)
        elapsed_time = time.time() - start_time
        scheduler.step()

        logger.info('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t score={:.6f} \t time={:.2f}s'.format(
            epoch + 1, epochs, avg_loss, avg_val_loss, score, elapsed_time))
        torch.save(model.state_dict(),
                   log_folder + answer_output_model_path.format(str(fold+1), str(epoch+1)))


    del train_df, val_df, model, optimizer, criterion#, scheduler
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
