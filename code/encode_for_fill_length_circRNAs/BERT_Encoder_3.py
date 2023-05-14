import numpy as np
import os.path as path
import torch
from transformers import BertModel, BertTokenizer
import os
import gc
from tempfile import mkdtemp
from path import BERT_MODEL_PATH, CIRC_RNA_DATA_PATH
import torch_xla
import torch_xla.core.xla_model as xm

# assert os.environ['COLAB_TPU_ADDR']
# dev = xm.xla_device()

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
            seq_len = (attention_mask[seq_num] == 1).sum()           
            seq_emd = embedding[seq_num][1:seq_len-1]
            seq_emd = np.array(seq_emd)
            # print(np.shape(seq_emd))
            # count += 1
            # print(count)
            # print(seq_emd)
            seq_emd = seq_emd.reshape((101,12,64,-1)).mean(axis=1).mean(2)
            features.append(seq_emd) 
        
    return features
    

def RNABert(line): 
    seq = line.strip() + 'AA'
    sequences = []
    sequences.append(seq2kmer(seq, 3))
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=1, shuffle=False)
    # print(np.shape(sequences))
    # print(np.shape(dataloader))
    Features = RNA_Bert(sequences, dataloader)
    # print(np.shape(Features[0]))
    # print(Features)
    return  np.array(Features[0])