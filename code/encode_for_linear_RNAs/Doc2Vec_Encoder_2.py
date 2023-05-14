import numpy as np
import gensim
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import sys
sys.path.insert(1, '/content/drive/MyDrive/CRIECNN/code/')
from path import DOC2VEC_MODEL_PATH, CIRC_RNA_DATA_PATH

def Doc2Vec_Embedding(protein):
    model = DOC2VEC_MODEL_PATH    
    seqpos_path = CIRC_RNA_DATA_PATH + protein + '/positive.txt'
    seqneg_path = CIRC_RNA_DATA_PATH + protein + '/negative.txt'
    seqpos = read_fasta_file(seqpos_path)
    seqneg = read_fasta_file(seqneg_path)
    X, y, embedding_matrix = RNA2Vec(10, 1, 30, model, 101, seqpos, seqneg)
    # print(np.shape(X))
    # print(np.shape(X[0]))
    # print((X[0]))
    # print(np.shape(X[1]))
    # print((X[1]))


    return X, y, embedding_matrix

def read_fasta_file(fasta_file):
    # seq_dict = {}
    # bag_sen = list()
    # fp = open(fasta_file, 'r')
    # name = ''
    # for line in fp:
    #     line = line.rstrip()
    #     if line[0]=='>': 
    #         name = line[1:] 
    #         seq_dict[name] = ''
    #     else:
    #         seq_dict[name] = seq_dict[name] + line.upper()
    # fp.close()    
    # for seq in seq_dict.values():
    #     seq = seq.replace('T', 'U')
    #     bag_sen.append(seq)
        
    # return np.asarray(bag_sen)

    seq_list = []
    f = open(fasta_file,'r')
    for line in f:
        if '>' not in line and 'N' not in line:
            line = line.strip().upper()
            # line = line + 'AA'
            seq_list.append(line)
    return seq_list
    

def RNA2Vec(k, s, vector_dim, model, MAX_LEN, pos_sequences, neg_sequences):
    model1 = gensim.models.Doc2Vec.load(model)
    pos_list = seq2ngram(pos_sequences, k, s, model1.wv)
    neg_list = seq2ngram(neg_sequences, k, s, model1.wv)
    seqs = pos_list + neg_list

    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    y = np.array([1] * len(pos_list) + [0] * len(neg_list))
    y = to_categorical(y)
    
    embedding_matrix = np.zeros((len(model1.wv.vocab), vector_dim))
    for i in range(len(model1.wv.vocab)):
        embedding_vector = model1.wv[model1.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector                
    return X, y, embedding_matrix

def seq2ngram(seqs, k, s, wv):
    list01 = []
    for num, line in enumerate(seqs):
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