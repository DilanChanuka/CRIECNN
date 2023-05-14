import numpy as np
import os.path as path
import torch
from transformers import BertModel, BertTokenizer
import os
import gc
from tempfile import mkdtemp
import sys
sys.path.insert(1, '/content/drive/MyDrive/CRIECNN/code/')
from path import BERT_MODEL_PATH, CIRC_RNA_DATA_PATH
import torch_xla
import torch_xla.core.xla_model as xm

# assert os.environ['COLAB_TPU_ADDR']
# dev = xm.xla_device()

def read_fasta(file_path):
    seq_list = []
    f = open(file_path,'r')
    for line in f:
        if '>' not in line and 'N' not in line:
            line = line.strip().upper()
            line = line + 'AA'
            seq_list.append(line)
    return seq_list

def seq2kmer(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def RNA_Bert(sequences, dataloader):
    features = []
    seq = []    
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, do_lower_case=False)
    model = BertModel.from_pretrained(BERT_MODEL_PATH, ignore_mismatched_sizes=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Assign TPU
    # device = dev
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model = model.eval()
    count = 0

    for sequences in dataloader:
        seq.append(sequences)    
        # print('sequences')
        # print(np.shape(sequences))
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        token_type_ids = torch.tensor(ids['token_type_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)[0]
        embedding = embedding.cpu().numpy()
        # print(np.shape(embedding))
        # count += 1
        # print(count)
        # print(len(embedding))
        for seq_num in range(len(embedding)):
            # print('seq_num')
            # print(seq_num)
            seq_len = (attention_mask[seq_num] == 1).sum()           
            seq_emd = embedding[seq_num][1:seq_len-1]
            seq_emd = np.array(seq_emd)
            
            # print(np.shape(seq_emd))
            # count += 1
            # print(count)

            seq_emd = seq_emd.reshape((101,12,64,-1)).mean(axis=1).mean(2)
            features.append(seq_emd) 
        # break
    return features
    
def makeArray(a):
    filename = path.join(mkdtemp(), 'tmp.txt')
    fp = np.memmap(filename, dtype='float32', mode='w+', shape=(len(a),101,64))
    fp[::] = a[::]
    fp.flush()
    newfp = np.memmap(filename, dtype='float32', mode='r', shape=(len(a),101,64))
    os.remove(filename)
    return newfp

def RNABert(protein):
    file_positive_path = CIRC_RNA_DATA_PATH + protein + '/positive.txt'
    file_negative_path = CIRC_RNA_DATA_PATH + protein + '/negative.txt'
    sequences_pos = read_fasta(file_positive_path)
    sequences_neg = read_fasta(file_negative_path)
    sequences_ALL = sequences_pos + sequences_neg
    sequences = []
    for seq in sequences_ALL:
        seq = seq.strip()
        seq_parser = seq2kmer(seq, 3)
        sequences.append(seq_parser)
 
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=100, shuffle=False)
    # print(np.shape(sequences))
    # print(np.shape(dataloader))

    Features = RNA_Bert(sequences, dataloader)
    data = makeArray(Features)
    gc.collect()
    return  data