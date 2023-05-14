import numpy as np
import os.path as path
import torch
from transformers import BertModel, BertTokenizer
import os
import gc
from tempfile import mkdtemp

import sys
sys.path.insert(1, '/content/drive/MyDrive/CRIECNN/code/')

from path import BERT_MODEL_PATH, CIRC_RNA_DATA_PATH, START_INDEX, END_INDEX

import torch_xla
import torch_xla.core.xla_model as xm

assert os.environ['COLAB_TPU_ADDR']
dev = xm.xla_device()

def read_fasta(file_path, seq_len):
    seq_list = []
    f = open(file_path,'r')
    content = f.readlines()
    firstset = content[START_INDEX:END_INDEX]
    for line in firstset:
        if '>' not in line:
            line = line.strip().upper()
            line = line.ljust(seq_len + 2, 'A')
            seq_list.append(line)
    return seq_list

def seq2kmer(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def RNA_Bert(sequences, dataloader, seq_length):
    features = []
    seq = []    
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, do_lower_case=False)
    model = BertModel.from_pretrained(BERT_MODEL_PATH, ignore_mismatched_sizes=True)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Assign TPU
    device = dev
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model = model.eval()
    for sequences in dataloader:
        seq.append(sequences)    
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        token_type_ids = torch.tensor(ids['token_type_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)[0]
        embedding = embedding.cpu().numpy()
    
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()           
            seq_emd = embedding[seq_num][1:seq_len-1]
            seq_emd = np.array(seq_emd)
            seq_emd = seq_emd.reshape((seq_length,12,64,-1)).mean(axis=1).mean(2)
            features.append(seq_emd) 
        
    return features
    
def makeArray(a, seq_len):
    filename = path.join(mkdtemp(), 'tmp.txt')
    fp = np.memmap(filename, dtype='float32', mode='w+', shape=(len(a),seq_len,64))
    fp[::] = a[::]
    fp.flush()
    newfp = np.memmap(filename, dtype='float32', mode='r', shape=(len(a),seq_len,64))
    os.remove(filename)
    return newfp

def RNABert(protein, seq_len):
    file_positive_path = CIRC_RNA_DATA_PATH + 'sequence_pos/' + protein + '_pos' 
    file_negative_path = CIRC_RNA_DATA_PATH + 'sequence_neg/' + protein + '_neg' 
    sequences_pos = read_fasta(file_positive_path, seq_len)
    sequences_neg = read_fasta(file_negative_path, seq_len)
    sequences_ALL = sequences_pos + sequences_neg
    sequences = []
    for seq in sequences_ALL:
        seq = seq.strip()
        seq_parser = seq2kmer(seq, 3)
        sequences.append(seq_parser)
 
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=100, shuffle=False)
    Features = RNA_Bert(sequences, dataloader, seq_len)
    data = makeArray(Features, seq_len)
    gc.collect()
    return  data

# if __name__ == "__main__":
#     data = RNABert("WTAP", 68)
#     print(len(data))
#     print(np.shape(data))
#     print(data)
