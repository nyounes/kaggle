import os
import numpy as np

import tensorflow_hub as hub

from logger import logger


def create_embedding_features(model_url, train, test, input_columns):

    # load universal sentence encoder model to get sentence ambeddings
    embed = hub.load(model_url)

    # create empty dictionaries to store final results
    embedding_train = {}
    embedding_test = {}

    # iterate over text columns to get senteces embeddings with the previous loaded model
    for text in input_columns:
        train_text = train[text].str.replace('?', '.').str.replace('!', '.').tolist()
        test_text = test[text].str.replace('?', '.').str.replace('!', '.').tolist()

        # create empy list to save each batch
        curr_train_emb = []
        curr_test_emb = []

        # define a batch to transform senteces to their correspinding embedding (1 X 512 for each sentece)
        batch_size = 4
        ind = 0
        while ind * batch_size < len(train_text):
            curr_train_emb.append(embed(train_text[ind * batch_size: (ind + 1) * batch_size])['outputs'].numpy())
            ind += 1

        ind = 0
        while ind * batch_size < len(test_text):
            curr_test_emb.append(embed(test_text[ind * batch_size: (ind + 1) * batch_size])['outputs'].numpy())
            ind += 1

        embedding_train[text + '_embedding'] = np.vstack(curr_train_emb)
        embedding_test[text + '_embedding'] = np.vstack(curr_test_emb)

    return embedding_train, embedding_test


def check_saved_embeddings(input_columns, embedding_folder):

    for text in input_columns:
        for data in ['train_', 'test_']:
            if not os.path.exists(embedding_folder + data + text + '_embedding.npy'):
                return False
    return True


def get_embedding_features(model_url, train, test, input_columns, save_embeddings=True,
                           embedding_folder="datas/inputs/", load_if_exists=True):

    if load_if_exists and check_saved_embeddings(input_columns, embedding_folder):
        logger.info("loading universal sentence encoder saved embeddings")
        embedding_train = {}
        embedding_test = {}
        for text in input_columns:
            embedding_train[text + '_embedding'] = np.load(embedding_folder + 'train_' + text + '_embedding.npy')
            embedding_test[text + '_embedding'] = np.load(embedding_folder + 'test_' + text + '_embedding.npy')
        return embedding_train, embedding_test

    logger.info("creating universal sentence encoder embeddings")
    embedding_train, embedding_test = create_embedding_features(model_url, train, test, input_columns)

    if save_embeddings:
        logger.info("saving universal sentence encoder embeddings")
        for text in input_columns:
            np.save(embedding_folder + 'train_' + text + '_embedding.npy', embedding_train[text + '_embedding'])
            np.save(embedding_folder + 'test_' + text + '_embedding.npy', embedding_test[text + '_embedding'])

    return embedding_train, embedding_test


def get_dist_features(embedding_train, embedding_test):

    def l2_dist(x, y):
        return np.power(x - y, 2).sum(axis=1)

    def cos_dist(x, y):
        return (x * y).sum(axis=1)

    dist_features_train = np.array([
        l2_dist(embedding_train['question_title_embedding'], embedding_train['answer_embedding']),
        l2_dist(embedding_train['question_body_embedding'], embedding_train['answer_embedding']),
        l2_dist(embedding_train['question_body_embedding'], embedding_train['question_title_embedding']),
        cos_dist(embedding_train['question_title_embedding'], embedding_train['answer_embedding']),
        cos_dist(embedding_train['question_body_embedding'], embedding_train['answer_embedding']),
        cos_dist(embedding_train['question_body_embedding'], embedding_train['question_title_embedding'])
    ]).T

    dist_features_test = np.array([
        l2_dist(embedding_test['question_title_embedding'], embedding_test['answer_embedding']),
        l2_dist(embedding_test['question_body_embedding'], embedding_test['answer_embedding']),
        l2_dist(embedding_test['question_body_embedding'], embedding_test['question_title_embedding']),
        cos_dist(embedding_test['question_title_embedding'], embedding_test['answer_embedding']),
        cos_dist(embedding_test['question_body_embedding'], embedding_test['answer_embedding']),
        cos_dist(embedding_test['question_body_embedding'], embedding_test['question_title_embedding'])
    ]).T

    return dist_features_train, dist_features_test
