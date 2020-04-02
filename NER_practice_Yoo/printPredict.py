import numpy as np
import copy
import os
import argparse
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from data_utils import Vocabulary
from data_utils import load_data_interactive

from data_loader import prepare_sequence, prepare_char_sequence, prepare_lex_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from CNN_BiLSTM import CNNBiLSTM
from data_loader import get_loader
from sklearn.metrics import f1_score

num_layers=2
embed_size=100
hidden_size=200
gpu_index=0

lex_cnt = 0

if lex_cnt == 33:
    predict_NER_dict = {0: '<PAD>',
                        1: 'B-PS_PROF', 2: 'B-PS_ENT', 3: 'B-PS_POL', 4: 'B-PS_NAME',
                        5: 'B-AF_REC', 6: 'B-AF_WARES', 7: 'B-AF_ITEM', 8: 'B-AF_SERVICE', 9: 'B-AF_OTHS',
                        10: 'B-OG_PRF', 11: 'B-OG_PRNF', 12: 'B-OG_PBF', 13: 'B-OG_PBNF',
                        14: 'B-LC_CNT', 15: 'B-LC_PLA', 16: 'B-LC_ADD', 17: 'B-LC_OTHS',
                        18: 'B-CV_TECH', 19: 'B-CV_LAWS', 20: 'B-EV_LT', 21: 'B-EV_ST',
                        22: 'B-GR_PLOR', 23: 'B-GR_PLCI', 24: 'B-TM_FLUC', 25: 'B-TM_ECOFIN', 26: 'B-TM_FUNC',
                        27: 'B-TM_CURR', 28: 'B-TM_OTHS', 29: 'B-PD_PD', 30: 'B-TI_TIME',
                        31: 'B-NUM_PRICE', 32: 'B-NUM_PERC', 33: 'B-NUM_OTHS',
                        34: 'I', 35: 'O'}

    NER_idx_dic = {'<unk>': 0, 'PS_PROF': 1, 'PS_ENT': 2, 'PS_POL': 3, 'PS_NAME': 4,
                   'AF_REC': 5, 'AF_WARES': 6, 'AF_ITEM': 7, 'AF_SERVICE': 8, 'AF_OTHS': 9,
                   'OG_PRF': 10, 'OG_PRNF': 11, 'OG_PBF': 12, 'OG_PBNF': 13,
                   'LC_CNT': 14, 'LC_PLA': 15, 'LC_ADD': 16, 'LC_OTHS': 17,
                   'CV_TECH': 18, 'CV_LAWS': 19, 'EV_LT': 20, 'EV_ST': 21,
                   'GR_PLOR': 22, 'GR_PLCI': 23, 'TM_FLUC': 24, 'TM_ECOFIN': 25, 'TM_FUNC': 26,
                   'TM_CURR': 27, 'TM_OTHS': 28, 'PD_PD': 29, 'TI_TIME': 30,
                   'NUM_PRICE': 31, 'NUM_PERC': 32, 'NUM_OTHS': 33,
                   'I': 34, 'O': 35}

else:
    predict_NER_dict = {0: '<PAD>', 1: 'B-PS', 2: 'B-AF',
                        3: 'B-OG', 4: 'B-LC', 5: 'B-CV',
                        6: 'B-EV', 7: 'B-GR', 8: 'B-TM',
                        9: 'B-PD', 10: 'B-TI', 11: 'B-NUM',
                        12: 'I', 13: 'O'}

    NER_idx_dic = {'<unk>': 0, 'PS': 1, 'AF': 2, 'OG': 3, 'LC': 4, 'CV': 5, 'EV': 6,
                   'GR': 7, 'TM': 8, 'PD': 9, 'TI': 10, 'NUM': 11, 'I': 12, 'O': 13}


# train.py에서 torch.save(model, PATH)를 통해 model이 저장되었음을 전제로 한다.
device = torch.device("cuda")

cnn_bilstm_tagger = CNNBiLSTM()

cnn_bilstm_tagger.load_state_dict(torch.load('PATH'))
cnn_bilstm_tagger.to(device)

cnn_bilstm_tagger.eval()


def preprocessing(x_text_batch, x_pos_batch, x_split_batch):
    x_text_char_item = []
    for x_word in x_text_batch[0]: # 첫번째 문장 반복
        x_char_item = []
        for x_char in x_word: #
            x_char_item.append(x_char)
        x_text_char_item.append(x_char_item)
    x_text_char_batch = [x_text_char_item]

    x_idx_item = prepare_sequence(x_text_batch[0], vocab.word2idx)
    x_idx_char_item = prepare_char_sequence(x_text_char_batch[0], char_vocab.word2idx)
    x_pos_item = prepare_sequence(x_pos_batch[0], pos_vocab.word2idx)
    x_lex_item = prepare_lex_sequence(x_text_batch[0], lex_dict)

    x_idx_batch = [x_idx_item]
    x_idx_char_batch = [x_idx_char_item]
    x_pos_batch = [x_pos_item]
    x_lex_batch = [x_lex_item]

    max_word_len = int(
        np.amax([len(word_tokens) for word_tokens in x_idx_batch]))  # ToDo: usually, np.mean can be applied
    batch_size = len(x_idx_batch)
    batch_words_len = [len(word_tokens) for word_tokens in x_idx_batch]
    batch_words_len = np.array(batch_words_len)

    # Padding procedure (word)
    padded_word_tokens_matrix = np.zeros((batch_size, max_word_len), dtype=np.int64)
    for i in range(padded_word_tokens_matrix.shape[0]):
        for j in range(padded_word_tokens_matrix.shape[1]):
            try:
                padded_word_tokens_matrix[i, j] = x_idx_batch[i][j]
            except IndexError:
                pass

    max_char_len = int(np.amax([len(char_tokens) for word_tokens in x_idx_char_batch for char_tokens in word_tokens]))
    if max_char_len < 5:  # size of maximum filter of CNN
        max_char_len = 5

    # Padding procedure (char)
    padded_char_tokens_matrix = np.zeros((batch_size, max_word_len, max_char_len), dtype=np.int64)
    for i in range(padded_char_tokens_matrix.shape[0]):
        for j in range(padded_char_tokens_matrix.shape[1]):
            for k in range(padded_char_tokens_matrix.shape[1]):
                try:
                    padded_char_tokens_matrix[i, j, k] = x_idx_char_batch[i][j][k]
                except IndexError:
                    pass

    # Padding procedure (pos)
    padded_pos_tokens_matrix = np.zeros((batch_size, max_word_len), dtype=np.int64)
    for i in range(padded_pos_tokens_matrix.shape[0]):
        for j in range(padded_pos_tokens_matrix.shape[1]):
            try:
                padded_pos_tokens_matrix[i, j] = x_pos_batch[i][j]
            except IndexError:
                pass

    # Padding procedure (lex)
    padded_lex_tokens_matrix = np.zeros((batch_size, max_word_len, len(NER_idx_dic)))
    for i in range(padded_lex_tokens_matrix.shape[0]):
        for j in range(padded_lex_tokens_matrix.shape[1]):
            for k in range(padded_lex_tokens_matrix.shape[2]):
                try:
                    for x_lex in x_lex_batch[i][j]:
                        k = NER_idx_dic[x_lex]
                        padded_lex_tokens_matrix[i, j, k] = 1
                except IndexError:
                    pass

    x_text_batch = x_text_batch
    x_split_batch = x_split_batch
    padded_word_tokens_matrix = torch.from_numpy(padded_word_tokens_matrix)
    padded_char_tokens_matrix = torch.from_numpy(padded_char_tokens_matrix)
    padded_pos_tokens_matrix = torch.from_numpy(padded_pos_tokens_matrix)
    padded_lex_tokens_matrix = torch.from_numpy(padded_lex_tokens_matrix).float()
    lengths = batch_words_len

    return x_text_batch, x_split_batch, padded_word_tokens_matrix, padded_char_tokens_matrix, padded_pos_tokens_matrix, padded_lex_tokens_matrix, lengths

