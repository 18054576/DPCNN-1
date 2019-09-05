from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from utils import *
import torch
import torch.nn as nn
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors


# read csv data to DataSet
dataset_train = DataSet.read_csv(train_csv,headers=('label','title','description'),sep='","')
dataset_test = DataSet.read_csv(test_csv,headers=('label','title','description'),sep='","')


# preprocess data
dataset_train.apply(lambda x: int(x['label'][1])-1,new_field_name='label')
dataset_train.apply(lambda x: x['title'].lower(), new_field_name='title')
dataset_train.apply(lambda x: x['description'][:-2].lower()+' .', new_field_name='description')

dataset_test.apply(lambda x: int(x['label'][1])-1,new_field_name='label')
dataset_test.apply(lambda x: x['title'].lower(), new_field_name='title')
dataset_test.apply(lambda x: x['description'][:-2].lower()+ ' .', new_field_name='description')

# split sentence with space
def split_sent(instance):
    return instance['description'].split()

dataset_train.apply(split_sent,new_field_name='description_words')
dataset_test.apply(split_sent,new_field_name='description_words')

# add item of length of words
dataset_train.apply(lambda x: len(x['description_words']),new_field_name='description_seq_len')
dataset_test.apply(lambda x: len(x['description_words']),new_field_name='description_seq_len')

# get max_sentence_length
max_seq_len_train=0
max_seq_len_test=0
for i in range (len(dataset_train)):
    if(dataset_train[i]['description_seq_len'] > max_seq_len_train):
        max_seq_len_train = dataset_train[i]['description_seq_len']
    else:
        pass
for i in range (len(dataset_test)):
    if(dataset_test[i]['description_seq_len'] > max_seq_len_test):
        max_seq_len_test = dataset_test[i]['description_seq_len']
    else:
        pass

max_sentence_length = max_seq_len_train
if (max_seq_len_test > max_sentence_length):
    max_sentence_length = max_seq_len_test
print ('max_sentence_length:',max_sentence_length)

# set input,which will be used in forward
dataset_train.set_input("description_words")
dataset_test.set_input("description_words")

# set target，which will be used in evaluation
dataset_train.set_target("label")
dataset_test.set_target("label")

# build vocabulary
vocab = Vocabulary(min_freq=2)
dataset_train.apply(lambda x:[vocab.add(word) for word in x['description_words']])
vocab.build_vocab()

# import glove embedding
tmp_file = get_tmpfile(tmp_path)
wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
vocab_size = len(vocab) + 1
embed_size = word_embedding_dimension
weight = torch.zeros(vocab_size+1, embed_size)

for i in range(len(wvmodel.index2word)):
    try:
        index = vocab.word2idx[wvmodel.index2word[i]]
    except:
        continue
    weight[index, :] = torch.from_numpy(wvmodel.get_vector(
        vocab.idx2word[vocab.word2idx[wvmodel.index2word[i]]]))
    

# index sentence by Vocabulary
dataset_train.apply(lambda x: [vocab.to_index(word) for word in x['description_words']],new_field_name='description_words')
dataset_test.apply(lambda x: [vocab.to_index(word) for word in x['description_words']],new_field_name='description_words')

# pad title_words to max_sentence_length
def padding_words(data):
    for i in range(len(data)):
        if data[i]['description_seq_len'] <= max_sentence_length:
            padding = [0] * (max_sentence_length - data[i]['description_seq_len'])
            data[i]['description_words'] += padding
        else:
            pass
    return data

dataset_train= padding_words(dataset_train)
dataset_test = padding_words(dataset_test)
dataset_train.apply(lambda x: len(x['description_words']), new_field_name='description_seq_len')
dataset_test.apply(lambda x: len(x['description_words']), new_field_name='description_seq_len')

dataset_train.rename_field("description_words","description_word_seq")
dataset_train.rename_field("label","label_seq")
dataset_test.rename_field("description_words","description_word_seq")
dataset_test.rename_field("label","label_seq")

print("dataset processed successfully!")
