import numpy as np
import os

# from konlpy.tag import Kkma
# from konlpy.tag import Twitter
from eunjeon import Mecab

from collections import Counter
import pickle
import codecs
import argparse


import re


mecab = Mecab()


def load_data_and_labels_exo(data_file_dir):
    # Load data_in from files
    x_mor_list = list()
    x_pos_list = list()
    x_split_list = list()
    y_list = list()

    file_obj = codecs.open(data_file_dir, "r", "utf-8")
    lines = file_obj.readlines()

    NER_label_list = [':PS', ':DT', ':LC', ':OG', ':TI']
    NER_dict = {'<PAD>': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'B_LC': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                'B_DT': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                'B_OG': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                'B_TI': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'B_PS': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                'I': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                'O': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    re_word = re.compile('<(.+?):[A-Z]{2}>')

    for line in lines:
        line = line.strip()

        raw_data = line.replace('<', '').replace('>', '').replace(':PS', '').replace(':DT', '').replace(':LC',
                                                                                                        '').replace(
            ':OG', '').replace(':TI', '')
        split_raw_data = raw_data.split(' ')
        pos_data = mecab.pos(raw_data)

        x_split = []
        x_mor = []
        x_pos = []

        i = 0
        len_pos_word = 0
        len_split_word = 0
        for mor_pos in pos_data:
            if mor_pos[0] in split_raw_data[i]:
                len_pos_word += len(mor_pos[0])
                len_split_word = len(split_raw_data[i])

                # new_pos_data.append([i, pos_word[0], pos_word[1]])

                x_split.append(i)
                x_mor.append(mor_pos[0])
                x_pos.append(mor_pos[1])

                if len_pos_word == len_split_word:
                    i = i + 1
                    len_pos_word = 0
                    len_split_word = 0

        if len(x_mor) == 0:  # mecab에러인지... 가끔 하나가 빠짐 그거 제외
            continue

        x_mor_list.append(x_mor)
        x_pos_list.append(x_pos)
        x_split_list.append(x_split)

        # label data
        label_data = line
        label_split_data = label_data.split(' ')

        re_result = re_word.finditer(label_data)
        raw_re_word_list = []
        temp_re_word_list = []
        re_NER_list = []
        for re_result_item in re_result:
            re_NER_list.append(re_result_item.group()[-3:-1])
            raw_re_word_list.append(re_word.findall((re_result_item.group())))
            temp_re_word_list.append(re_word.findall((re_result_item.group()[1:])))
            for i, temp_re_word_item in enumerate(temp_re_word_list):
                if len(temp_re_word_item) != 0:
                    raw_re_word_list[i] = temp_re_word_item

        # re_NER_list = re_NER.findall(label_data)
        re_word_list = [[re_word[0].replace(' ', '')] for re_word in raw_re_word_list]
        # print("re_word_list:",re_word_list)

        y_data = ['O'] * len(x_mor)
        B_flag = 0
        data_len = 0
        B_I_data_len = 0

        for i in range(len(x_mor)):
            pos_i_split = x_split[i]
            word_mor = x_mor[i]
            pos = x_pos[i]

            if len(re_word_list) == 0:
                continue

            if word_mor in re_word_list[0][0]:

                # print("word_mor:", word_mor)
                # print("data_len:", data_len)
                # print("B_I_data_len:", B_I_data_len)

                if B_flag == 0 and re_word_list[0][0].startswith(word_mor):

                    data_len += len(word_mor)
                    B_I_data_len = len(re_word_list[0][0])

                    y_data[i] = 'B_' + re_NER_list[0]
                    B_flag = 1  # B_ token mark

                    if data_len == B_I_data_len:
                        re_word_list.pop(0)
                        re_NER_list.pop(0)

                        data_len = 0
                        B_I_data_len = 0
                        B_flag = 0  # B_ token mark init


                    elif i + 1 < len(x_mor):
                        if x_mor[i + 1] not in re_word_list[0][0]:  # 시작일줄 알았는데 서브스트링이고, 매칭도 안되고 다음글자가 속하지 않으면 다시 리셋
                            y_data[i] = 'O'
                            B_flag = 0
                            data_len = 0
                            B_I_data_len = 0
                            B_flag = 0  # B_ token mark init


                elif B_flag == 1:

                    data_len += len(word_mor)
                    B_I_data_len = len(re_word_list[0][0])

                    if data_len != B_I_data_len:
                        y_data[i] = 'I'
                    elif data_len == B_I_data_len:
                        y_data[i] = 'I'
                        re_word_list.pop(0)
                        re_NER_list.pop(0)
                        data_len = 0
                        B_I_data_len = 0
                        B_flag = 0

        # print("y_data: ", y_data)
        y_data_idx = []
        for y in y_data:
            y_data_idx.append(NER_dict[y])
        y_list.append(y_data_idx)

    y_list = np.array(y_list)

    return x_mor_list, x_pos_list, x_split_list, y_list


def load_data_and_labels_klp(data_file_dir):

    # Load data_in from files
    x_mor_list = list()
    x_pos_list = list()
    x_split_list = list()
    y_list = list()


    file_obj = codecs.open(data_file_dir, "r", "utf-8" )
    lines = file_obj.readlines()

    NER_label_list = [':PS',':DT',':LC',':OG',':TI']
    NER_dict = {'<PAD>': [1, 0, 0, 0, 0, 0, 0, 0],
                'B_LC': [0, 1, 0, 0, 0, 0, 0, 0],
                'B_DT': [0, 0, 1, 0, 0, 0, 0, 0],
                'B_OG': [0, 0, 0, 1, 0, 0, 0, 0],
                'B_TI': [0, 0, 0, 0, 1, 0, 0, 0],
                'B_PS': [0, 0, 0, 0, 0, 1, 0, 0],
                'I': [0, 0, 0, 0, 0, 0, 1, 0],
                'O': [0, 0, 0, 0, 0, 0, 0, 1]}

    re_word = re.compile('<(.+?):[A-Z]{2}>')

    for line in lines:
        line = line.strip() #좌우 공백 제거

        if len(line) == 0:
            continue

        elif line[0] == ';': # raw data
            raw_data = line.replace('; ','')
            split_raw_data = raw_data.split(' ')
            pos_data = mecab.pos(raw_data)

            x_split = []
            x_mor = []
            x_pos = []

            i = 0
            len_pos_word = 0
            len_split_word = 0
            for mor_pos in pos_data:
                if mor_pos[0] in split_raw_data[i]:
                    len_pos_word += len(mor_pos[0])
                    len_split_word = len(split_raw_data[i])

                    # new_pos_data.append([i, pos_word[0], pos_word[1]])

                    x_split.append(i)
                    x_mor.append(mor_pos[0])
                    x_pos.append(mor_pos[1])



                    if len_pos_word == len_split_word:
                        i = i + 1
                        len_pos_word = 0
                        len_split_word = 0

            if len(x_mor) == 0:  # mecab에러인지... 가끔 하나가 빠짐 그거 제외
                continue

            x_mor_list.append(x_mor)
            x_pos_list.append(x_pos)
            x_split_list.append(x_split)

            # print("x_mor", x_mor)


        elif line[0] == '$': # label data
            label_data = line.replace('$','')
            # print("label_data: ",label_data)
            label_split_data = label_data.split(' ')

            re_result = re_word.finditer(label_data)
            raw_re_word_list = []
            temp_re_word_list = []
            re_NER_list = []
            for re_result_item in re_result:
                re_NER_list.append(re_result_item.group()[-3:-1])
                raw_re_word_list.append(re_word.findall((re_result_item.group())))
                temp_re_word_list.append(re_word.findall((re_result_item.group()[1:])))
                for i, temp_re_word_item in enumerate(temp_re_word_list):
                    if len(temp_re_word_item) != 0:
                        raw_re_word_list[i] = temp_re_word_item


            # re_NER_list = re_NER.findall(label_data)
            re_word_list = [[re_word[0].replace(' ', '')] for re_word in raw_re_word_list]
            # print("re_word_list:",re_word_list)

            y_data = ['O'] * len(x_mor)
            B_flag = 0
            data_len = 0
            B_I_data_len = 0


            for i in range(len(x_mor)):
                pos_i_split = x_split[i]
                word_mor = x_mor[i]
                pos = x_pos[i]

                if len(re_word_list) == 0:
                    continue

                if word_mor in re_word_list[0][0]:


                    # print("word_mor:", word_mor)
                    # print("data_len:", data_len)
                    # print("B_I_data_len:", B_I_data_len)

                    if B_flag == 0 and re_word_list[0][0].startswith(word_mor):

                        data_len += len(word_mor)
                        B_I_data_len = len(re_word_list[0][0])

                        y_data[i] = 'B_' + re_NER_list[0]
                        B_flag = 1  # B_ token mark

                        if data_len == B_I_data_len:
                            re_word_list.pop(0)
                            re_NER_list.pop(0)

                            data_len = 0
                            B_I_data_len = 0
                            B_flag = 0  # B_ token mark init

                        elif i+1 < len(x_mor):
                            if x_mor[i + 1] not in re_word_list[0][0]:  # 시작일줄 알았는데 서브스트링이고, 매칭도 안되고 다음글자가 속하지 않으면 다시 리셋
                                y_data[i] = 'O'
                                B_flag = 0
                                data_len = 0
                                B_I_data_len = 0
                                B_flag = 0  # B_ token mark init


                    elif B_flag == 1:

                        data_len += len(word_mor)
                        B_I_data_len = len(re_word_list[0][0])

                        if data_len != B_I_data_len:
                            y_data[i] = 'I'
                        elif data_len == B_I_data_len:
                            y_data[i] = 'I'
                            re_word_list.pop(0)
                            re_NER_list.pop(0)
                            data_len = 0
                            B_I_data_len = 0
                            B_flag = 0

            # print("y_data: ", y_data)
            y_data_idx = []
            for y in y_data:
                y_data_idx.append(NER_dict[y])
            y_list.append(y_data_idx)


    #y_list = np.array(y_list)


    return x_mor_list, x_pos_list, x_split_list, y_list


x_mor_list, x_pos_list, x_split_list, y_list = load_data_and_labels_klp('data_in/my_test')

# x_mor_list, x_pos_list, x_split_list, y_list = load_data_and_labels_exo('data_in/EXOBRAIN_NE_CORPUS_10000.txt')

# x_mor_list, x_pos_list, x_split_list, y_list = load_data_and_labels_exo('data/labeled/2012_reviewed_4_labeled.txt')

NER_dict = {'<PAD>': [1, 0, 0, 0, 0, 0, 0, 0],
                'B_LC': [0, 1, 0, 0, 0, 0, 0, 0],
                'B_DT': [0, 0, 1, 0, 0, 0, 0, 0],
                'B_OG': [0, 0, 0, 1, 0, 0, 0, 0],
                'B_TI': [0, 0, 0, 0, 1, 0, 0, 0],
                'B_PS': [0, 0, 0, 0, 0, 1, 0, 0],
                'I': [0, 0, 0, 0, 0, 0, 1, 0],
                'O': [0, 0, 0, 0, 0, 0, 0, 1]}

print(x_mor_list)


# y_idx_list = []
# for i in y_list:
#     y_idx=[]
#     for j in i:
#         idx = np.argmax(j)
#         y_idx.append(idx)
#     y_idx_list.append(y_idx)
#
# print(y_idx_list)


# y_ner_list=[]
# for i in y_list:
#     y_ner=[]
#     for j in i:
#         for ner, value in NER_dict.items():
#             if j==value:
#                 y_ner.append(ner)
#     y_ner_list.append(y_ner)
#
# ner_set = []
# for i in y_ner_list:
#     k = set(i)
#     ner_set.append(k)
#
# print(ner_set)
