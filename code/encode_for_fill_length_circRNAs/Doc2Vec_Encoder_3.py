import numpy as np
import gensim
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from path import DOC2VEC_MODEL_PATH, CIRC_RNA_DATA_PATH

def Doc2Vec_Embedding(line):
    model = DOC2VEC_MODEL_PATH     
    X, embedding_matrix = RNA2Vec(10, 1, 30, model, 101, line)
    # print(np.shape(X))
    # print(np.shape(X[0]))
    # print((X[0]))
    # print(np.shape(X[1]))
    # print((X[1]))

    return np.array(X[0])

def RNA2Vec(k, s, vector_dim, model, MAX_LEN, pos_sequences):
    model1 = gensim.models.Doc2Vec.load(model)
    pos_list = seq2ngram(pos_sequences, k, s, model1.wv)
    seqs = pos_list
    # print('Doc2Vec: ' + str(np.shape(seqs)))
    # print(seqs)
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')    
    # print(str(np.shape(seqs)))
    embedding_matrix = np.zeros((len(model1.wv.vocab), vector_dim))
    for i in range(len(model1.wv.vocab)):
        embedding_vector = model1.wv[model1.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector                
    return X, embedding_matrix

def seq2ngram(seqs, k, s, wv):
    list01 = []
    # print(np.shape(seqs))
    # print(seqs)
    # for num, line in enumerate(seqs):
    num = 0
    line = seqs
    # print(str(num) + ' **** ' + line)
    if num < 3000000:
        line = line.strip()
        l = len(line) 
        list2 = []
        for i in range(0, l, s):
            if i + k >= l + 1:
                break
            list2.append(line[i:i + k])
        list01.append(convert_data_to_index(list2, wv))
    return list01

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data