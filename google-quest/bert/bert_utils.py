from tqdm import tqdm
from math import floor, ceil
import numpy as np

import torch

TARGET_COLS_DICT = {
    'question_asker_intent_understanding': "[Q_INTENT]",
    'question_body_critical': "[Q_CRITICAL]",
    'question_conversational': "[Q_CONVERSATIONAL]",
    'question_expect_short_answer': "[Q_SHORT]",
    'question_fact_seeking': "[Q_FACT]",
    'question_has_commonly_accepted_answer': "[Q_ACCEPTED]",
    'question_interestingness_others': "[Q_OTHERS]",
    'question_interestingness_self': "[Q_SELF]",
    'question_multi_intent': "[Q_MULTI_INTENT]",
    'question_not_really_a_question': "[Q_NOTAQUESTION]",
    'question_opinion_seeking': "[Q_OPINION]",
    'question_type_choice': "[Q_CHOICE]",
    'question_type_compare': "[Q_COMPARE]",
    'question_type_consequence': "[Q_CONSEQUENCE]",
    'question_type_definition': "[Q_DEFINITION]",
    'question_type_entity': "[Q_ENTITY]",
    'question_type_instructions': "[Q_INSTRUCTIONS]",
    'question_type_procedure': "[Q_PROCEDURE]",
    'question_type_reason_explanation': "[Q_EXPLANATION]",
    'question_type_spelling': "[Q_SPELLING]",
    'question_well_written': "[Q_WELLWRITTEN]",
    'answer_helpful': "[A_HELPFUL]",
    'answer_level_of_information': "[A_INFORMATION]",
    'answer_plausible': "[A_PLAUSIBLE]",
    'answer_relevance': "[A_RELEVANCE]",
    'answer_satisfaction': "[A_SATISFACTION]",
    'answer_type_instructions': "[A_INSTRUCTIONS]",
    'answer_type_procedure': "[A_PROCEDURE]",
    'answer_type_reason_explanation': "[A_EXPLANATION]",
    'answer_well_written': "[A_WELLWRITTEN]"
}


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""

    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")

    segments = []
    first_sep = True
    current_segment_id = 0

    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


def _trim_input(tokenizer, title, question, answer, max_sequence_length=512,
                t_max_len=30, q_max_len=239, a_max_len=239):

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)

    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:

        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len

        if t_new_len + a_new_len + q_new_len + 4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d" % (
                max_sequence_length, (t_new_len + a_new_len + q_new_len + 4)))

        t = t[:t_new_len]

        q_len_head = round(q_new_len/2)
        q_len_tail = -1* (q_new_len -q_len_head)
        a_len_head = round(a_new_len/2)
        a_len_tail = -1* (a_new_len -a_len_head)
        q = q[:q_len_head]+q[q_len_tail:]
        a = a[:a_len_head]+a[a_len_tail:]

        # q = q[:q_new_len]
        # a = a[:a_new_len]

    return t, q, a


def _convert_to_bert_inputs_question_wlabels(title, question, answer, labels, label_columns, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    aux_tokens = []
    for label in label_columns:
        if 'question' in label:
            if labels[label] > 0:
                aux_tokens.append(TARGET_COLS_DICT[label])

    stoken = ["[CLS]"] + aux_tokens + ["[SEP]"] + title + ["[SEP]"] + question + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def _convert_to_bert_inputs_answer_wlabels(title, question, answer, labels, label_columns, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    aux_tokens = []
    for label in label_columns:
        if 'answer' in label:
            if labels[label] > 0:
                aux_tokens.append(TARGET_COLS_DICT[label])

    stoken = ["[CLS]"] + aux_tokens + ["[SEP]"] + title + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def _convert_to_bert_inputs_question(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def _convert_to_bert_inputs_answer(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["[CLS]"] + title + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arays(df, columns, tokenizer, max_sequence_length):

    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(tokenizer, t, q, a, max_sequence_length)
        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    ]


def compute_input_arrays_question_wlabels(df, columns, label_columns, tokenizer, max_sequence_length):

    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df.iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(tokenizer, t, q, a, max_sequence_length)
        ids, masks, segments = _convert_to_bert_inputs_question_wlabels(
            t, q, a, instance[label_columns], label_columns, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    ]


def compute_input_arrays_answer_wlabels(df, columns, label_columns, tokenizer, max_sequence_length):

    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df.iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(tokenizer, t, q, a, max_sequence_length)
        ids, masks, segments = _convert_to_bert_inputs_answer_wlabels(
            t, q, a, instance[label_columns], label_columns, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    ]


def compute_input_arrays_question(df, columns, tokenizer, max_sequence_length):

    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(tokenizer, t, q, a, max_sequence_length,
                              t_max_len=60, q_max_len=448, a_max_len=0)
        ids, masks, segments = _convert_to_bert_inputs_question(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    ]


def compute_input_arrays_answer(df, columns, tokenizer, max_sequence_length):

    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(tokenizer, t, q, a, max_sequence_length,
                              t_max_len=60, q_max_len=0, a_max_len=448)
        ids, masks, segments = _convert_to_bert_inputs_answer(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    ]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])
