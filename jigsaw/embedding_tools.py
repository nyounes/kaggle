import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from logger import logger


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(file):
    return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(open(file)))


def build_single_embedding(word_index, path, vector_size, save_matrix, matrix_path, load_if_exists=True):
    if load_if_exists:
        try:
            logger.info("Loading matrix from {}".format(matrix_path))
            embedding_matrix = np.load(matrix_path)
            return embedding_matrix
        except Exception as e:
            logger.info("No matrix found")

    logger.info("Building matrix from {}".format(path))
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, vector_size))
    unknown_words = []

    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)

    np.save(matrix_path, embedding_matrix)
    return embedding_matrix


def build_embedding_matrix(embedding_list, tokenizer, config):
    embedding_matrix_inputs = []
    for embedding_input in embedding_list:
        embedding_matrix_inputs.append(
            build_single_embedding(
                tokenizer.word_index,
                os.path.join(config['input_path'], config[embedding_input]['embedding_path']), 300,
                save_matrix=True, matrix_path=os.path.join(config['input_path'], config[embedding_input]['matrix_path'])
            )
        )

    embedding_matrix = np.concatenate(embedding_matrix_inputs, axis=-1)

    return embedding_matrix


def add_caps_feature(word_index):
    caps_feature = np.zeros((len(word_index) + 1, 1))
    for word, i in word_index.items():
        if len(word) > 1 and word.isupper():
            caps_feature[i] = 1
        else:
            caps_feature[i] = 0
    return caps_feature


def add_bad_words_feature(word_index, bad_words_path):
    bad_words = pd.read_csv(bad_words_path)
    bad_words = list(bad_words['words'].values)
    bad_words_feature = np.zeros((len(word_index) + 1, 1))

    for word, i in word_index.items():
        if word.lower() in bad_words:
            bad_words_feature[i] = 1
        else:
            bad_words_feature[i] = 0
    return bad_words_feature
