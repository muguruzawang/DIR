import sys 
sys.path.append("..")

import torch
import dgl
import numpy as np
import json
import pickle
import random
from itertools import combinations
from collections import Counter
from tools.logger import *
from utils.logging import init_logger, logger
from module import vocabulary
from module import data
from torch.autograd import Variable
from collections import Counter
from nltk.corpus import stopwords
import nltk
import re
from transformers import  AutoTokenizer, AutoModel

import time

import pdb

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)

def load_to_cuda(batch,device):
    batch = {'input': batch['input'].to(device, non_blocking=True), 'summaries':batch['summaries'].to(device, non_blocking=True), 'cands':batch['cands'], 'rank_index':batch['rank_index'].to(device, non_blocking=True)}
    return batch 

def readJson(fname):
    data = []
    with open(fname, 'r',encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def readText(fname):
    data = []
    with open(fname, 'r', encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

def write_txt(batch, seqs, w_file, args):
    # converting the prediction to real text.
    ret = []
    for b, seq in enumerate(seqs):
        txt = []
        for token in seq:
            if int(token) not in [args.wordvocab.word2id(x) for x in ['<PAD>', '<BOS>', '<EOS>']]:
                txt.append(args.wordvocab.id2word(int(token)))
            if int(token) == args.wordvocab.word2id('<EOS>'):
                break
        w_file.write(' '.join([str(x) for x in txt])+'\n')
        ret.append([' '.join([str(x) for x in txt])])
    return ret 


def replace_ent(x, ent, V):
    # replace the entity
    mask = x>=V
    if mask.sum()==0:
        return x
    nz = mask.nonzero()
    fill_ent = ent[nz, x[mask]-V]
    x = x.masked_scatter(mask, fill_ent)
    return x


###这是建立长度为lens的mask矩阵的操作
def len2mask(lens, device):
    #得到最大的长度n
    max_len = max(lens)
    #构造维度为[len(lens),maxlen]的矩阵
    mask = torch.arange(max_len, device=device).unsqueeze(0).expand(len(lens), max_len)
    ####最终会得到类似[[ 0, 0, 0，1，1],
        #[0, 0, 1, 1, 1],
        #[0, 1, 1, 1, 1]]的矩阵
    #作者这里用0来表示实际的单词，用1来填充
    mask = mask >= torch.LongTensor(lens).to(mask).unsqueeze(1)
    return mask


### for roberta, use pad_id = 1 to pad tensors.
def pad(var_len_list, out_type='list', flatten=False):
    if flatten:
        lens = [len(x) for x in var_len_list]
        var_len_list = sum(var_len_list, [])
    max_len = max([len(x) for x in var_len_list])
    if out_type=='list':
        if flatten:
            return [x+['<PAD>']*(max_len-len(x)) for x in var_len_list], lens
        else:
            return [x+['<PAD>']*(max_len-len(x)) for x in var_len_list]
    if out_type=='tensor':
        if flatten:
            return torch.stack([torch.cat([x, \
            torch.ones([max_len-len(x)]+list(x.shape[1:])).type_as(x)], 0) for x in var_len_list], 0), lens
        else:
            return torch.stack([torch.cat([x, \
            torch.ones([max_len-len(x)]+list(x.shape[1:])).type_as(x)], 0) for x in var_len_list], 0)

def pad_sent_entity(var_len_list, pad_id,bos_id,eos_id, flatten=False):
    def _pad_(data,height,width,pad_id,bos_id,eos_id):
        rtn_data = []
        for para in data:
            if torch.is_tensor(para):
                para = para.numpy().tolist()
            if len(para) > width:
                para = para[:width]
            else:
                para += [pad_id] * (width - len(para))
            rtn_data.append(para)
        rtn_length = [len(para) for para in data]
        x = []
        '''
        x.append(bos_id)
        x.append(eos_id)
        '''
        x.extend([pad_id] * (width))
        rtn_data = rtn_data + [x] * (height - len(data))
        # rtn_data = rtn_data + [[pad_id] * width] * (height - len(data))
        rtn_length = rtn_length + [0] * (height - len(data))
        if len(rtn_data) == 0:
            rtn_data.append([])
        return rtn_data, rtn_length
    
    if flatten:
        var_len = [len(x) for x in var_len_list]
        max_nsent = max(var_len)
        max_ntoken = max([max([len(p) for p in x]) for x in var_len_list])
        _pad_var_list = [_pad_(ex, max_nsent, max_ntoken, pad_id, bos_id, eos_id) for ex in var_len_list]
        pad_var_list = torch.stack([torch.tensor(e[0]) for e in _pad_var_list])
        return pad_var_list, var_len

    else:
        max_nsent = len(var_len_list)
        max_ntoken = max([len(x) for x in var_len_list])
        
        _pad_var_list = _pad_(var_len_list, max_nsent,max_ntoken, pad_id, bos_id, eos_id)
        pad_var_list = torch.tensor(_pad_var_list[0])
        return pad_var_list

def pad_edges(batch_example):
    max_nsent = max([len(ex.raw_sent_input) for ex in batch_example])
    max_nent = max([len(ex.raw_ent_text) for ex in batch_example])
    edges = torch.zeros(len(batch_example),max_nsent,max_nent)
    for index,ex in enumerate(batch_example):
        for key in ex.entities:
            if int(key) >= ex.doc_max_len:
                break
            if ex.entities[key] != []:
                for x in ex.entities[key]:
                    #e = at_least(x.lower().split())
                    e = at_least(x.lower())
                    entNo = ex.raw_ent_text.index(e)
                    sentNo = int(key)

                    edges[index][sentNo][entNo] = 1
    return edges

def at_least(x):
    # handling the illegal data
    if len(x) == 0:
        return ['<UNK>']
    else:
        return x

class Example(object):
    def __init__(self, target, references, summary, candidate_summaries, rank_index, sent_max_len, doc_max_len, docset_max_len, tokenizer):
        self.tokenizer = tokenizer
        
        # self.sep_vid = self.tokenizer.vocab[self.sep_token]
        # self.cls_vid = self.tokenizer.vocab[self.cls_token]
        # self.pad_vid = self.tokenizer.vocab[self.pad_token]

        self.cands = candidate_summaries
        self.summaries = []
        for i in rank_index:
            tgt_ori = candidate_summaries[i]
            tgt_ori = tgt_ori.lower()
            tgt_ori_split = tgt_ori.split()
            if len(tgt_ori_split) > 300:
                tgt_ori_split = tgt_ori_split[:300]
                tgt_ori = ' '.join(tgt_ori_split)
            tgt_ori = re.sub(r'@cite\_\d{1,3}','@cite',tgt_ori)

            token = self.tokenizer.encode(tgt_ori)

            self.summaries.append(token)

        self.source = []
        self.source.append(target)
        for key in references:
            ref = references[key]['abstract']
            if ref != "":
                self.source.append(ref)

        self.source = ' '.join(self.source)
        
        source_tokens = self.tokenizer.encode(self.source)

        if len(source_tokens) > 4096:
            source_tokens = source_tokens[:4095]+ [source_tokens[-1]]

        self.source_tokens = source_tokens        
        self.rank_index = rank_index

    def __str__(self):
        return '\n'.join([str(k)+':\t'+str(v) for k, v in self.__dict__.items()])

    def __len__(self):
        return len(self.raw_text)


class ExampleSet(torch.utils.data.Dataset):
    def __init__(self, text_path, tokenizer, sent_max_len, doc_max_len, docset_max_len, device=None, training = True):
        super(ExampleSet, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.training = training

        self.pad_token = '<pad>'
        self.begin_token = '<s>'
        self.end_token = '</s>'

        self.bosid = self.tokenizer.vocab[self.begin_token]
        self.eosid = self.tokenizer.vocab[self.end_token]
        self.padid = self.tokenizer.vocab[self.pad_token]

        self.sent_max_len = sent_max_len
        self.doc_max_len = doc_max_len
        self.docset_max_len = docset_max_len

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.json_text_list = readJson(text_path) ###将训练数据读出来（text: , summary: ,label:）
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.json_text_list))
        self.size = len(self.json_text_list) ###训练集的大小

    def get_example(self, index):
        json_text = self.json_text_list[index]
        #e["summary"] = e.setdefault("summary", [])
        example = Example(json_text['target_paper'], json_text['reference'], json_text['summary'], json_text['candidate_summaries'], json_text['rank_index'], self.sent_max_len, self.doc_max_len, self.docset_max_len, self.tokenizer)
        return example
    
    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_len, :self.doc_max_len]
        N, m = label_m.shape
        if m < self.doc_max_len:
            pad_m = np.zeros((N, self.doc_max_len - m))
            return np.hstack([label_m, pad_m])
        return label_m

    def __getitem__(self, index):
        item = self.get_example(index)
        #enc_sent_input_pad是包含所有经过pad后的句子列表，这一步是对句子进行裁剪，只取前max个句子
        ex_data = self.get_tensor(item)
        return ex_data

    def __len__(self):
        return self.size

    def get_tensor(self, ex):
        _cached_tensor = {'input':torch.LongTensor(ex.source_tokens), 'summaries': [torch.LongTensor(x) for x in ex.summaries], 'cands':ex.cands, 'rank_index':torch.LongTensor(ex.rank_index)}
        return _cached_tensor

    def batch_fn(self, samples):
        batch_input, batch_summaries, batch_cands, batch_rank_index = [],[],[],[]
        #batch_ex = map(list, zip(*samples))
        for ex_data in samples:
            if ex_data != {}:
                batch_input.append(ex_data['input'])
                batch_summaries.append(ex_data['summaries'])
                batch_cands.append(ex_data['cands'])
                batch_rank_index.append(ex_data['rank_index'])
        
        batch_input = pad_sent_entity(batch_input, self.padid,self.bosid,self.eosid, flatten = False)
        batch_summaries,_ = pad_sent_entity(batch_summaries, self.padid,self.bosid,self.eosid, flatten = True)

        batch_rank_index = torch.stack(batch_rank_index, dim=0)
        
        return {'input': batch_input, 'summaries':batch_summaries, 'cands':batch_cands, 'rank_index':batch_rank_index}
        
if __name__ == '__main__' :
    pass

