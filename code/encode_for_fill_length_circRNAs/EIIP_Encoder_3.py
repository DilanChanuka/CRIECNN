import numpy as np
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

def RNA_EIIP(line):     
    dataX = encode(line.strip())          
    return np.array(dataX)