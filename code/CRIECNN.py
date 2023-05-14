import argparse

from Doc2Vec_Encoder import Doc2Vec_Embedding
from Kmer_Encoder import RNA_Kmer 
from BERT_Encoder import RNABert
from EIIP_Encoder import RNA_EIIP

import sys
# sys.path.insert(1, '/content/drive/MyDrive/CRIECNN/code/encode_for_3db_combination/')

# from Doc2Vec_Encoder_1 import Doc2Vec_Embedding
# from Kmer_Encoder_1 import RNA_Kmer 
# from BERT_Encoder_1 import RNABert
# from EIIP_Encoder_1 import RNA_EIIP

from path import RESULT_PATH
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
# from protein_list import circRNA_RBPs, circRNA_seq_len
from protein_list import circRNA_RBPs
import time

gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=tf_config)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
np.random.seed(4)

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
#Connect to the TPU handle and initialise it
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

def CRIECNN(args):
    protein = args.protein
    storage = args.storage
    user = args.user
    # seq_len = circRNA_seq_len[protein]
    
    Kmer_features = RNA_Kmer(protein)
    Embedding, dataY,  embedding_matrix = Doc2Vec_Embedding(protein)
    BERT_features = RNABert(protein)
    EIIP_features = RNA_EIIP(protein)

    print(np.shape(Kmer_features))
    print(np.shape(Embedding))
    print(np.shape(BERT_features))
    print(np.shape(EIIP_features))
    
    indexes = np.random.choice(Kmer_features.shape[0],Kmer_features.shape[0], replace=False)    
    training_idx, test_idx = indexes[:round(((Kmer_features.shape[0])/10)*8)], indexes[round(((Kmer_features.shape[0])/10)*8):]
        
    train_sequence, test_sequence = Kmer_features[training_idx, :, :], Kmer_features[test_idx, :, :]
    train_profile, test_profile = Embedding[training_idx, :], Embedding[test_idx, :]
    train_property, test_property = BERT_features[training_idx, :, :], BERT_features[test_idx, :, :]
    train_main, test_main = EIIP_features[training_idx, :, :], EIIP_features[test_idx, :, :]

    train_label, test_label = dataY[training_idx, :], dataY[test_idx, :]       

    if len(dataY) >= 20000:
        batchSize = 1000
    elif len(dataY) >= 10000:
        batchSize = 500
    else:
        batchSize = 50

    maxEpochs = 100
    basic_path = storage + user + '/'
    methodName = protein
    
    logging.basicConfig(level=logging.DEBUG)
    sys.stdout = sys.stderr
    logging.debug("Loading data...")          
    print("Number of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))    
    
    tprs=[]
    mean_fpr=np.linspace(0,1,100)
        
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    aucs = []
    Accs = []
    precisions = []
    recalls = []
    fscores = []
    time_diffs = []
    i = 0
    
    for train_index, eval_index in kf.split(train_label):
        
        train_X1 = train_sequence[train_index]
        train_X2 = train_profile[train_index]
        train_X3 = train_property[train_index]  
        train_X4 = train_main[train_index]
        train_y = train_label[train_index]

        eval_X1 = train_sequence[eval_index]
        eval_X2 = train_profile[eval_index]
        eval_X3 = train_property[eval_index]
        eval_X4 = train_main[eval_index]
        eval_y = train_label[eval_index]        

        [MODEL_PATH, CHECKPOINT_PATH, LOG_PATH, RESULT_PATH] = defineExperimentPaths(basic_path, methodName, str(i))
        logging.debug("Loading network/training configuration...")
        
        # Assign TPU
        with strategy.scope():    
            model = createECNN(embedding_matrix)
            logging.debug("Model summary ... ")
            model.count_params()
            # model.summary()
            checkpoint_weight = CHECKPOINT_PATH + "weights.best.hdf5"
            if (os.path.exists(checkpoint_weight)):
                print ("load previous best weights")
                model.load_weights(checkpoint_weight)
            model.compile(optimizer='adam', loss={'ss_output': 'categorical_crossentropy'},metrics = ['accuracy'])
            logging.debug("Running training...")
            
            def step_decay(epoch):
                initial_lrate = 0.0005
                drop = 0.8
                epochs_drop = 5.0            
                lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
                # print ('Learning Rate : ' + str(lrate))
                return lrate
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto'),
                ModelCheckpoint(checkpoint_weight,
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='auto',
                                period=1),
                LearningRateScheduler(step_decay),
            ]
            
            T1 = time.time()        
            history = model.fit(
                {'sequence_input': train_X1, 'profile_input': train_X2, 'property_input': train_X3, 'main_input': train_X4 },
                {'ss_output': train_y},
                epochs=maxEpochs,
                batch_size=batchSize,
                callbacks=callbacks,
                verbose = 1,
                validation_data=(
                    {'sequence_input': eval_X1, 'profile_input': eval_X2, 'property_input': eval_X3, 'main_input': eval_X4},
                    {'ss_output':eval_y}),
                shuffle=True)     

            T2 = time.time()    
            time_diff =  T2 - T1    
            time_diffs.append(time_diff)
            
            logging.debug("Saving final model...")
            model.save(os.path.join(MODEL_PATH, 'model.h5'), overwrite=True)
            json_string = model.to_json()
            with open(os.path.join(MODEL_PATH, 'model.json'), 'w') as f:
                f.write(json_string)

            logging.debug("Make prediction...")       
            prediction_result = model.predict(
                {'sequence_input': test_sequence, 'profile_input': test_profile, 'property_input': test_property, 'main_input': test_main}) 

            ytrue = test_label[:, 1]
            ypred = prediction_result[:, 1]
            
            y_pred = np.argmax(prediction_result, axis=-1)
            auc = roc_auc_score(ytrue, ypred)
            fpr,tpr,thresholds=roc_curve(ytrue,ypred)
            tprs.append(interp(mean_fpr,fpr,tpr))
            aucs.append(auc)        
            acc = accuracy_score(ytrue, y_pred)
            Accs.append(acc)
            precision = precision_score(ytrue, y_pred)  
            recall = recall_score(ytrue, y_pred)
            fscore = f1_score(ytrue, y_pred)         
            precisions.append(precision)
            recalls.append(recall)
            fscores.append(fscore)     
            i = i + 1
        
        print(protein+' Fold AUC: '+ str(auc) +'\nFold ACC: '+ str(acc) + '\nFold Precision: '
              + str(precision) + '\nFold Recall: '+ str(recall) + '\nFold F1 Score: ' + str(fscore) + '\nFold Train Time: ' + str(time_diff))
    
    mean_tpr=np.mean(tprs,axis=0)
    mean_tpr[-1]=1.0
    mean_auc= np.mean(aucs)
    mean_acc = np.mean(Accs)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_fscore = np.mean(fscores)
    mean_time_diff = np.mean(time_diffs)
    
    print(protein+" acid AUC: %.4f " % mean_auc)
    print(protein+" acid ACC: %.4f " % mean_acc)
    print(protein+" acid Precision: %.4f " % mean_precision)
    print(protein+" acid Recall: %.4f " % mean_recall)
    print(protein+" acid F1 Score: %.4f " % mean_fscore)
    print(protein+" mean Training time: %.4f " % mean_time_diff)
                    
    random_color= (random.random(), random.random(), random.random())
    plt.plot(mean_fpr, mean_tpr, color=random_color, lw=1, label= protein+': %0.4f' % mean_auc)
    # save_result_files(storage, protein, mean_fpr, mean_tpr, 
    #                   mean_auc, mean_acc, mean_precision, mean_recall, mean_fscore, mean_time_diff)
    
    return mean_auc, mean_time_diff
        

def save_result_files(storage, protein, mean_fpr, mean_tpr, 
                      mean_auc, mean_acc, mean_precision, mean_recall, mean_fscore, mean_time_diff):
    
    path = storage+'AUC_data/'+protein+'/'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, 'mean_fpr.txt' ), 'w') as temp_file:
        temp_file.write(str(mean_fpr))

    with open(os.path.join(path, 'mean_tpr.txt' ), 'w') as temp_file:
        temp_file.write(str(mean_tpr))

    with open(os.path.join(path, 'AUC.txt' ), 'w') as temp_file:
        temp_file.write(str(mean_auc))
        
    with open(os.path.join(path, 'all_measures.txt' ), 'w') as temp_file:
        temp_file.write('AUC ' + str(mean_auc) + '\n' + 'ACC ' + str(mean_acc) + '\n' 
                        + 'Precision ' + str(mean_precision) + '\n' + 'Recall ' + str(mean_recall) + '\n' 
                        + 'F1-Score ' + str(mean_fscore) + '\n' + 'Time ' + str(mean_time_diff))
        
        
def parse_arguments(parser):
    parser.add_argument('--protein', type = str,  default = 'WTAP')
    parser.add_argument('--storage', type = str, default = RESULT_PATH)
    parser.add_argument('--user', type = str, default = 'admin')
    args = parser.parse_args()
    return args   

def defineExperimentPaths(basic_path, methodName, experimentID):
    experiment_name = methodName + '/' + experimentID
    MODEL_PATH = basic_path + experiment_name + '/model/'
    LOG_PATH = basic_path + experiment_name + '/logs/'
    CHECKPOINT_PATH = basic_path + experiment_name + '/checkpoints/'
    RESULT_PATH = basic_path + experiment_name + '/results/'
    mk_dir(MODEL_PATH)
    mk_dir(CHECKPOINT_PATH)
    mk_dir(RESULT_PATH)
    mk_dir(LOG_PATH)
    return [MODEL_PATH, CHECKPOINT_PATH, LOG_PATH, RESULT_PATH]

def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    plt.figure()    
    
    # CRIECNN(args)
        
    result = []
    AUCsList = []
    for p in circRNA_RBPs:
      oneProtein = []
      args.protein = p

      AUC, Time = CRIECNN(args)
      # AUC, Time = CRIECNN(args)
      # AUC, Time = CRIECNN(args)
      # AUC, Time = CRIECNN(args)
      # AUC, Time = CRIECNN(args)

      oneProtein.append(p)
      oneProtein.append(AUC)
      AUCsList.append(AUC)
      oneProtein.append(Time)
      result.append(oneProtein)
      print("Protein: {p}, Final AUC: {auc}, total time: {time}".format(p=oneProtein[0], auc=oneProtein[1], time=oneProtein[2]))

      plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver operating characteristic curve')
      plt.legend(loc="lower right")
      plt.show()

    print("Overall results *************************************************")
    cnt = 1
    for one in result:
      print("{cnt}. Protein: {p}, Final AUC: {auc}, total time: {time}".format(cnt=cnt, p=one[0], auc=one[1], time=one[2]))
      cnt += 1

    print("Overall Mean AUC for All datasets: ", np.mean(AUCsList))
