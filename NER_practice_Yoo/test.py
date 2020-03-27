from pprint import pprint

import pickle


with open('./data_in/vocab_ko_NER.pkl', 'rb') as f:
    vocab = pickle.load(f)
print("len(vocab): ", len(vocab))

print(vocab)