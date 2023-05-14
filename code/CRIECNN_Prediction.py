import argparse

# from Doc2Vec_Encoder import Doc2Vec_Embedding
# from Kmer_Encoder import RNA_Kmer 
# from BERT_Encoder import RNABert
# from EIIP_Encoder import RNA_EIIP

import sys
sys.path.insert(1, '/content/drive/MyDrive/CRIECNN/code/encode_for_fill_length_circRNAs/')

from Doc2Vec_Encoder_3 import Doc2Vec_Embedding
from Kmer_Encoder_3 import RNA_Kmer
from BERT_Encoder_3 import RNABert
from EIIP_Encoder_3 import RNA_EIIP

from path import RESULT_PATH, CIRC_RNA_DATA_PATH
from ECNN import createECNN
import numpy as np
import logging
import tensorflow as tf
import os
import sklearn
from sklearn.model_selection import KFold
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from scipy import interp
import matplotlib.pyplot as plt
import random
from protein_list import circRNA_RBPs
import time
from keras.models import load_model
from keras_self_attention import SeqSelfAttention

gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=tf_config)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
np.random.seed(4)

# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
# #Connect to the TPU handle and initialise it
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)

def CRIECNN(args):
    protein = args.protein
    predictmodel = args.predictType
    storage = args.storage
    user = args.user

    rootdir = CIRC_RNA_DATA_PATH + protein + '/'
    pathList = []
    fileList = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isfile(d):
            pathList.append(d)
            fileList.append(file)

    for circRNAFile in fileList:
        all_probs = []
        with open(rootdir + circRNAFile) as f:
          for line in f:
            line = line.strip()
            if len(line) >= 10: 
              if len(line) != 101:
                line = line.ljust(101, 'A')
              Kmer_features = RNA_Kmer(line)
              Embedding = Doc2Vec_Embedding(line)
              BERT_features = RNABert(line)
              EIIP_features = RNA_EIIP(line)
              f1 = []
              f2 = []
              f3 = []
              f4 = []
              f1.append(Kmer_features)
              f2.append(Embedding)
              f3.append(BERT_features)
              f4.append(EIIP_features) 
              f1 = np.array(f1)
              f2 = np.array(f2)
              f3 = np.array(f3)
              f4 = np.array(f4)

              # Assign TPU
              # with strategy.scope():
              modelPredict = load_model(predictmodel, custom_objects = {'SeqSelfAttention':SeqSelfAttention})             
              predictedResult = modelPredict.predict(
                {'profile_input': f2, 'property_input': f3, 'sequence_input': f1, 'main_input': f4})      
              predictedResult = predictedResult[:, 1]
              print(protein + ' -> ' + circRNAFile +' One Fragment Prediction Result: '+ str(predictedResult))
              save_result_files(storage, protein, circRNAFile, predictedResult[0])
              all_probs.append(predictedResult[0])
          save_avg_result_file(storage, protein, circRNAFile, np.mean(all_probs))
          print('***********************************************************')
          print('Mean Probability: ' + str(np.mean(all_probs)))


def save_result_files(storage, protein, circRNAFile, prob):    
    path = storage+'Full_length_fragments_probability_data/'+protein+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, circRNAFile+'.txt' ), 'a') as temp_file:
        temp_file.write(str(prob)+'\n')

def save_avg_result_file(storage, protein, circRNAFile, prob):    
    path = storage+'Full_length_fragments_probability_data/'+protein+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, circRNAFile+'_average.txt' ), 'w') as temp_file:
        temp_file.write(str(prob))

def parse_arguments(parser):
    parser.add_argument('--protein', type = str,  default = 'WTAP')
    parser.add_argument('--storage', type = str, default = RESULT_PATH)
    parser.add_argument('--user', type = str, default = 'admin')
    parser.add_argument('--predictType', type=str, default= RESULT_PATH +'admin/WTAP/4/model/model.h5')    
    args = parser.parse_args()
    return args   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)    
    for p in circRNA_RBPs:
      args.protein = p
      args.predictType = RESULT_PATH + 'admin/' + p + '/4/model/model.h5'
      CRIECNN(args)