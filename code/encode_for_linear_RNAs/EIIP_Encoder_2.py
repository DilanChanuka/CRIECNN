import numpy as np
import sys
sys.path.insert(1, '/content/drive/MyDrive/CRIECNN/code/')
from path import CIRC_RNA_DATA_PATH

nt = 'AGCT'
EIIP_dict =  {
                'A': 0.1260, 
                'G': 0.0806, 
                'C': 0.1340, 
                'T': 0.1335                          
             }

def encode(seq):
    vectors = np.zeros((len(seq), 4))
    for i in range(len(seq)):
        vectors[i][nt.index(seq[i])] = EIIP_dict[seq[i]]
    return vectors.tolist()

def RNA_EIIP(protein):   
    dataX = []
    with open(CIRC_RNA_DATA_PATH + protein + '/positive.txt') as f:
        for line in f:
            if '>' not in line and 'N' not in line:
                dataX.append(encode(line.strip()))              

    with open(CIRC_RNA_DATA_PATH + protein + '/negative.txt') as f:
        for line in f:
            if '>' not in line and 'N' not in line:
                dataX.append(encode(line.strip()))
                       
    return np.array(dataX)