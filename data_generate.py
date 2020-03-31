import re
import json
from eunjeon import Mecab
import pathlib
import torch
import numpy as np


# 작동원리
# 1. data_generator 함수에 json 파일을 입력하여 형태소 분석기(mecab)으로 분해한 형태소, 형태소종류, 그리고 개체명종류를 가져온다
# 2. 모델의 결과값으로 나온 정수값을 다시 값으로 매핑하기 위하여 {값:정수}, {정수:값} 의 형태를 가지는 딕셔너리를 만든다
# 3. 데이터를 모델에 넣기 위해 정수 인코딩한다
# 4. 문장간의 길이를 맞추기 위해 0으로 패딩한다

# get_data() 함수에 data type(train or test)만 넣으면 자동적으로 데이터를 만들 수 있다

def get_data(data_type):
    
    print('시간이 조금 오래 걸릴 수 있습니다.')
    # 1
    str_words = []
    pos = []
    lex = []

    json_paths = pathlib.Path('./json/json_' + data_type).glob('*')
    for json_path in json_paths:
        str_words += data_generator(json_path)[0]
        pos += data_generator(json_path)[1]
        lex += data_generator(json_path)[2]
    update_tag_scheme(lex)

    # 2
    word_dico, word_to_id, id_to_word = word_mapping(str_words)
    char_dico, char_to_id, id_to_char = char_mapping(str_words)
    pos_dico, pos_to_id, id_to_pos = pos_mapping(pos)
    lex_dico, lex_to_id, id_to_lex = lex_mapping(33)

    # 3
    encoded_words, encoded_char, encoded_pos, encoded_lex = encoding(str_words, pos, lex, word_to_id, char_to_id, pos_to_id, lex_to_id)

    # 4.
    padded_words, padded_char, padded_pos, padded_lex = padding(encoded_words, encoded_char, encoded_pos, encoded_lex, lex_to_id)

    # 
    labels = encoded_lex
    lengths = len(labels)
    
    
    print('data generate success!')
    return str_words, padded_words, padded_char, padded_pos, padded_lex, labels, lengths







#############################################################################################
#get_data() 함수 작동에 필요한 다른 함수들
#############################################################################################

# iob 태깅 타입 바꾸기
def update_tag_scheme(sentences, tag_scheme='iob'):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = s
        # Check that tags are given in the IOB format
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            new_tags = iob2(tags)
            for word, new_tag in zip(s, new_tags):
                word = new_tag
        else:
            raise Exception('Unknown tagging scheme!')

def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return tags



def data_generator(path):
    # json 라이브러리를 이용해 json 파일을 읽어온다
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    datas = json_data['data']  # 문장들만 가져오기
    attributes = json_data['attributes']['entities']['items']  # 단어의 정보를 담고 있는 딕셔너리들을 가져온다

    # attributes 순서를 전체 문장에서 나오는 순서로 바꾼다(start index 기준으로 오름차순 정렬)
    attributes = sorted(attributes, key=lambda x: x['mentions'][0]['startOffset'])

    # 함수

    # entity가 있는 시작위치를 얻을 수 있는 함수
    def get_start_index(i):
        return attributes[i]['mentions'][0]['startOffset']

    # entity가 있는 마지막 위치를 얻을 수 있는 함수
    def get_end_index(i):
        return attributes[i]['mentions'][0]['endOffset']

    # entity type(ex.PD_PD)를 얻을 수 있는 함수
    def get_entity(i):
        return attributes[i]['type']

    # 문서에 있는 모든 entity(문서에 나오는 순서로 저장되어 있다)
    entities = [get_entity(i) for i in range(len(attributes))]

    # entity인 단어
    entity_words = [datas[get_start_index(i):get_end_index(i)].strip().replace(' ', '') for i in
                    range(len(attributes))]

    sentences = re.split('\r\n|\u3000', datas)  # 정규표현식을 이용해 문서를 문장으로 나눈다
    sentences = [x for x in sentences if len(x) != 0]  # 길이가 0인 문장 제거

    ###### 형태소로 분해하기 ######
    # mecab
    mecab = Mecab()
    mecab_sentences = []  # mecab으로 형태소 분석 된 문장을 저장할 리스트

    for sentence in sentences:
        mecab_sentences.append(mecab.pos(sentence))

    # 모든 문장의 형태소에 대해 [pos_word, pos, entity] 형태로 저장할 리스트. 리스트는 한 문장 단위로 값이 들어간다.
    word_pos_entity_all = []
    all_word = []
    all_pos = []
    all_entity = []

    idx = 0  # entity로 취급되는 단어 리스트에서 단어를 가져오기 위해 사용하는 인덱스

    for mecab_sentence in mecab_sentences:  # 전체 문장

        # word_pos_entity = []
        w = []
        p = []
        e = []
        word_len = 0  # 형태소 분해된 단어의 길이
        entity_len = 0  # entity 단어의 길이

        for word, pos in mecab_sentence:  # 문장

            if word in entity_words[idx]:  # 형태소가 entity 단어에 포함되면
                # 형태소와 entity 단어의 길이를 잰다
                word_len += len(word)
                entity_len = len(entity_words[idx])
                # word_pos_entity.append([word, pos, 'I-' + entities[idx]])
                w.append(word)
                p.append(pos)
                e.append('I-'+entities[idx])

                # 형태소의 길이와 entity 단어의 길이가 같아지면 인덱스가 다음 entity 단어를 가리키게 +1을 해준다
                if word_len == entity_len:
                    word_len = 0
                    entity_len = 0
                    idx += 1
            # 형태소가 entity가 아닌 경우
            else:
                # word_pos_entity.append([word, pos, 'O'])
                w.append(word)
                p.append(pos)
                e.append('O')

        # word_pos_entity_all.append(word_pos_entity)
        all_word.append(w)
        all_pos.append(p)
        all_entity.append(e)
    # return word_pos_entity_all, all_word, all_pos, all_entity
    return all_word, all_pos, all_entity


# 딕셔너리 생성 함수
def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_tag_dico(item_list):

    dico = {}
    for item in item_list:
        if item not in dico:
            dico[item] = 1
        else:
            dico[item] += 1

    return dico


# mapping 함수들
def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def word_mapping(sentences, lower=1):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x.lower() if lower else x for x in s] for s in sentences]
    dico = create_dico(words)

    dico['<PAD>'] = 0
    dico['<UNK>'] = 10000000
    dico = {k:v for k,v in dico.items() if v>=3}
    word_to_id, id_to_word = create_mapping(dico)

    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word

def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    dico['<PAD>'] = 0
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char

def lex_mapping(lex_cnt=33):
    """
    Create a dictionary and a mapping of lexs
    """
    # tags = [[word[-1] for word in s] for s in sentences]
    # tags_11 = ['O', 'PS', 'AF', 'OG', 'LC', 'CV', 'GR', 'TM', 'PD', 'TI', 'NUM']
    # tags_33 = ['O', 'PS_PROF', 'PS_ENT', 'PS_POL', 'PS_NAME',
    #            'AF_REC', 'AF_WARES', 'AF_ITEM', 'AF_SERVICE', 'AF_OTHS',
    #            'OG_PRF', '']
    if lex_cnt == 33:
        lexs = ['<UNK>','B-PS_PROF','B-PS_ENT', 'B-PS_POL', 'B-PS_NAME', 'B-AF_REC',
                   'B-AF_WARES','B-AF_ITEM','B-AF_SERVICE','B-AF_OTHS','B-OG_PRF','B-OG_PRNF1','B-OG_PBF','B-OG_PBNF',
                   'B-LC_CNT','B-LC_PLA','B-LC_ADD','B-LC_OTHS','B-CV_TECH','B-CV_LAWS','B-EV_LT','B-EV_ST',
                   'B-GR_PLOR','B-GR_PLCI','B-TM_FLUC','B-TM_ECOFIN','B-TM_FUNC','B-TM_CURR','B-TM_OTHS','B-PD_PD',
                   'B-TI_TIME','B-NUM_PRICE', 'B-NUM_PERC', 'B-NUM_OTHS', 'I-PS_PROF', 'I-PS_ENT', 'I-PS_POL', 'I-PS_NAME', 'I-AF_REC',
                   'I-AF_WARES','I-AF_ITEM','I-AF_SERVICE','I-AF_OTHS','I-OG_PRF','I-OG_PRNF','I-OG_PBF', 'I-OG_PBNF',
                   'I-LC_CNT','I-LC_PLA','I-LC_ADD','I-LC_OTHS','I-CV_TECH','I-CV_LAWS','I-EV_LT','I-EV_ST',
                   'I-GR_PLOR','I-GR_PLCI','I-TM_FLUC','I-TM_ECOFIN','I-TM_FUNC','I-TM_CURR','I-TM_OTHS','I-PD_PD',
                   'I-TI_TIME','I-NUM_PRICE', 'I-NUM_PERC', 'I-NUM_OTHS', 'O']
    elif lex_cnt == 11:
        lexs = ['<UNK>', 'B-PS', 'B-AF', 'B-OG', 'B-LC', 'B-CV', 'B-EV', 'B-GR', 'B-TM', 'B-PD', 'B-TI', 'B-NUM',
                'I-PS', 'I-AF', 'I-OG', 'I-LC', 'I-CV', 'I-EV', 'I-GR', 'I-TM', 'I-PD', 'I-TI', 'I-NUM', 'O']

    dico = create_tag_dico(lexs)
    dico['<PAD>'] = 0
    lex_to_id, id_to_lex = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, lex_to_id, id_to_lex

def pos_mapping(sentences):
    poses = [[pos if '+' not in pos else '<UNK>' for pos in s] for s in sentences]
    poses = [['<UNK>' if pos=='UNKNOWN' else pos for pos in s] for s in poses]
    dico = create_dico(poses)
    dico['<PAD>'] = 0
    dico['<UNK>'] = 10000000
    pos_to_id, id_to_pos = create_mapping(dico)
    print('Found %i unique poses (%i in total)' % (len(dico), sum(len(x) for x in poses)))

    return dico, pos_to_id, id_to_pos


# 정수인코딩(만든 딕셔너리를 기반으로)
def encoding(str_words, pos, lex, word_to_id, char_to_id, pos_to_id, lex_to_id, lower=1):
    encoded_words = []
    encoded_char = []
    encoded_pos = []
    encoded_lex = []

    def f(x): return x.lower() if lower else x

    for s, p, l in zip(str_words, pos, lex):
        encoded_words.append([word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in s])

        # Skip characters that are not in the training set
        encoded_char.append([[char_to_id[c] for c in w if c in char_to_id]
             for w in s])

        encoded_pos.append([pos_to_id[pos] if pos in pos_to_id else pos_to_id['<UNK>'] for pos in p])

        encoded_lex.append([lex_to_id[lex] if lex in lex_to_id else lex_to_id['<UNK>'] for lex in l])

    print('encoding success!')
    return encoded_words, encoded_char, encoded_pos, encoded_lex

# 패딩
def padding(words, chars, poses, lexs, lex_to_id):
    size = len(words)
    max_len = max(len(sentence) for sentence in words)

    # Padding procedure (word)
    padded_word_tokens_matrix = np.zeros((size, max_len), dtype=np.int64)
    for i in range(padded_word_tokens_matrix.shape[0]):
        for j in range(padded_word_tokens_matrix.shape[1]):
            try:
                padded_word_tokens_matrix[i, j] = words[i][j]
            except IndexError:
                pass

    s = np.concatenate([sentence for sentence in chars])
    MAX_CHAR_LEN = max(np.concatenate([sentence for sentence in s]))
    if MAX_CHAR_LEN < 5:  # size of maximum filter of CNN
        MAX_CHAR_LEN = 5

    # Padding procedure (char)
    padded_char_tokens_matrix = np.zeros((size, max_len, MAX_CHAR_LEN), dtype=np.int64)
    for i in range(padded_char_tokens_matrix.shape[0]):
        for j in range(padded_char_tokens_matrix.shape[1]):
            for k in range(padded_char_tokens_matrix.shape[1]):
                try:
                    padded_char_tokens_matrix[i, j, k] = chars[i][j][k]
                except IndexError:
                    pass

    # Padding procedure (pos)
    padded_pos_tokens_matrix = np.zeros((size, max_len), dtype=np.int64)
    for i in range(padded_pos_tokens_matrix.shape[0]):
        for j in range(padded_pos_tokens_matrix.shape[1]):
            try:
                padded_pos_tokens_matrix[i, j] = poses[i][j]
            except IndexError:
                pass

    # Padding procedure (lex)
    # padded_lex_tokens_matrix = np.zeros((size, max_len, len(lex_to_id)))
    # for i in range(padded_lex_tokens_matrix.shape[0]):
    #     for j in range(padded_lex_tokens_matrix.shape[1]):
    #         for k in range(padded_lex_tokens_matrix.shape[2]):
    #             try:
    #                 for x_lex in lexs[i][j]:
    #                     k = lex_to_id[x_lex]
    #                     padded_lex_tokens_matrix[i, j, k] = 1
    #             except IndexError:
    #                 pass

    padded_lex_tokens_matrix = np.zeros((size, max_len), dtype=np.int64)
    for i in range(padded_lex_tokens_matrix.shape[0]):
        for j in range(padded_lex_tokens_matrix.shape[1]):
            try:
                padded_lex_tokens_matrix[i, j] = lexs[i][j]
            except IndexError:
                pass

    padded_word_tokens_matrix = torch.from_numpy(padded_word_tokens_matrix)
    padded_char_tokens_matrix = torch.from_numpy(padded_char_tokens_matrix)
    padded_pos_tokens_matrix = torch.from_numpy(padded_pos_tokens_matrix)
    padded_lex_tokens_matrix = torch.from_numpy(padded_lex_tokens_matrix)

    return padded_word_tokens_matrix, padded_char_tokens_matrix, padded_pos_tokens_matrix, padded_lex_tokens_matrix



