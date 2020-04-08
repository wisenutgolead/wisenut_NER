from data_utils import Vocabulary, build_vocab, build_char_vocab
import re
from eunjeon import Mecab

train_text_path = './data_in/wisenut_train.txt'

def make_word_vocab(train_text_path, threshold=3):
    with open(train_text_path, 'r', encoding='utf-8') as f:
        text_list = f.readlines() # 텍스트 파일의 모든 문장을 리스트에 담아 반환한다

    text_list = [text for text in text_list if text != '\n'] # 줄바꿈도 한 문장으로 취급되어 저장되어서 이를 삭제한다

    raw_compile = re.compile('<|(:.{2,3}>)') # 태그 제거하기 위한 정규표현식
    raw_text_list = [re.sub(raw_compile, '', text) for text in text_list] # 태그 제거한 원본 문장

    mecab = Mecab()
    mecab_text_list = [[word for word, pos in mecab.pos(text)] for text in raw_text_list]
    mecab_pos_list = [[pos for word, pos in mecab.pos(text)] for text in raw_text_list]

    word_vocab = build_vocab(mecab_text_list, threshold)
    print(word_vocab.word2idx)
    print('making word vocab success!')

def make_char_vocab(train_text_path, threshold):
    with open(train_text_path, 'r', encoding='utf-8') as f:
        text_list = f.readlines()  # 텍스트 파일의 모든 문장을 리스트에 담아 반환한다

    text_list = [text for text in text_list if text != '\n']  # 줄바꿈도 한 문장으로 취급되어 저장되어서 이를 삭제한다

    raw_compile = re.compile('<|(:.{2,3}>)')  # 태그 제거하기 위한 정규표현식
    raw_text_list = [re.sub(raw_compile, '', text) for text in text_list]  # 태그 제거한 원본 문장

    mecab = Mecab()
    mecab_text_list = [[word for word, pos in mecab.pos(text)] for text in raw_text_list]

    char_vocab = build_char_vocab(mecab_text_list, threshold)
    print(char_vocab.word2idx)
    print('making char vocab success!')

# def make_lex_dict():
#     lex_dict = {}
#     json_paths = pathlib.Path('./json/json_train').glob('*')
#     for json_path in json_paths:
#         with open(json_path, 'r', encoding='utf-8') as f:
#             json_data = json.load(f)
#
#         datas = json_data['data']  # 문장들만 가져오기
#         attributes = json_data['attributes']['entities']['items']  # 단어의 정보를 담고 있는 딕셔너리들을 가져온다
#
#         # attributes 순서를 전체 문장에서 나오는 순서로 바꾼다(start index 기준으로 오름차순 정렬)
#         attributes = sorted(attributes, key=lambda x: x['mentions'][0]['startOffset'])
#
#         # 함수
#
#         # entity가 있는 시작위치를 얻을 수 있는 함수
#         def get_start_index(i):
#             return attributes[i]['mentions'][0]['startOffset']
#
#         # entity가 있는 마지막 위치를 얻을 수 있는 함수
#         def get_end_index(i):
#             return attributes[i]['mentions'][0]['endOffset']
#
#         # entity type(ex.PD_PD)를 얻을 수 있는 함수
#         def get_entity(i):
#             return attributes[i]['type']
#
#         # 문서에 있는 모든 entity(문서에 나오는 순서로 저장되어 있다)
#         entities = [get_entity(i) for i in range(len(attributes))]
#
#         # entity인 단어
#         entity_words = [datas[get_start_index(i):get_end_index(i)].strip().replace(' ', '') for i in
#                         range(len(attributes))]
#
#         for word, entity in zip(entity_words, entities):
#             entity_5 = entity.split('_')[0]
#             if word not in lex_dict:
#                 lex_dict[word] = [entity]
#             else:
#                 entity_list = lex_dict[word]
#                 entity_list.append(entity)
#                 lex_dict[word] = list(set(entity_list))
#     print(lex_dict.keys())
#     print(lex_dict.values())
#     print('making lex dict success!')

def make_lex_dict(train_text_path):

    lex_dict = {}

    with open(train_text_path, 'r', encoding='utf-8') as f:
        text_list = f.readlines()  # 텍스트 파일의 모든 문장을 리스트에 담아 반환한다

    text_list = [text for text in text_list if text != '\n']  # 줄바꿈도 한 문장으로 취급되어 저장되어서 이를 삭제한다

    word_compile = re.compile('[<](.*?)[:]') # entity 처리가 되어있는 단어만 추출하는 정규표현식
    entity_compile = re.compile('[:](.{2})[>]') # entity만 추출하는 정규표현식

    for text in text_list:
        words = re.findall(word_compile, text)
        entities = re.findall(entity_compile, text)

        for word, entity in zip(words, entities):
            word = word.strip().replace(' ', '')
            if word not in lex_dict:
                lex_dict[word] = [entity]
            else:
                entity_list = lex_dict[word]
                entity_list.append(entity)
                lex_dict[word] = list(set(entity_list))

    print(lex_dict.keys())
    print(lex_dict.values())
    print('making lex dict success!')


