import numpy as np

import sys
sys.path.insert(1, '/content/drive/MyDrive/CRIECNN/code/')
from path import CIRC_RNA_DATA_PATH,START_INDEX, END_INDEX

nt = 'AGCT'
EIIP_dict =  {
                'A': 0.1260, 
                'G': 0.0806, 
                'C': 0.1340, 
                'T': 0.1335                          
             }

def encode(seq, seq_len):
    vectors = np.zeros((seq_len, 4))
    for i in range(len(seq)):
        vectors[i][nt.index(seq[i].upper())] = EIIP_dict[seq[i].upper()]
    return vectors.tolist()

def RNA_EIIP(protein, seq_len):   
    dataX = []
    with open(CIRC_RNA_DATA_PATH + 'sequence_pos/' + protein + '_pos') as f:
        content = f.readlines()
        firstset = content[START_INDEX:END_INDEX]
        for line in firstset:
            if '>' not in line:
                dataX.append(encode(line.strip(), seq_len))              

    with open(CIRC_RNA_DATA_PATH + 'sequence_neg/' + protein + '_neg') as f:
        content = f.readlines()
        firstset = content[START_INDEX:END_INDEX]
        for line in firstset:
            if '>' not in line:
                dataX.append(encode(line.strip(), seq_len))
                       
    return np.array(dataX)

# if __name__ == "__main__":
#     data = RNA_EIIP("WTAP", 68)
#     print(len(data))
#     print(np.shape(data))
#     print(data)