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

import time

import pdb

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)

def load_to_cuda(batch,device):
    batch = {'article_len':batch['article_len'],\
             'raw_tgt_text': batch['raw_tgt_text'], 'examples':batch['examples'], 'tgt': batch['tgt'].to(device, non_blocking=True), \
             'sent_num':batch['sent_num'], 'extra_zeros':batch['extra_zeros'], \
             'article_oovs':batch['article_oovs'],'tgt_extend': batch['tgt_extend'].to(device, non_blocking=True),\
             'text': batch['text'].to(device, non_blocking=True),'text_extend': batch['text_extend'].to(device,non_blocking=True),\
             'cand': batch['cand'].to(device, non_blocking=True),'cand_extend': batch['cand_extend'].to(device,non_blocking=True),\
             'bows':batch['bows'].to(device,non_blocking=True),'tgt_enc': batch['tgt_enc'].to(device,non_blocking=True),\
             'diffbows':batch['diffbows'].to(device,non_blocking=True) }
    batch['extra_zeros'] = batch['extra_zeros'].to(device, non_blocking=True) if batch['extra_zeros'] != None else batch['extra_zeros']
    
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
            data.append(eval(line.strip()))
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


###���������������������lens���mask���������������
def len2mask(lens, device):
    #���������������������n
    max_len = max(lens)
    #���������������[len(lens),maxlen]���������
    mask = torch.arange(max_len, device=device).unsqueeze(0).expand(len(lens), max_len)
    ####���������������������[[ 0, 0, 0���1���1],
        #[0, 0, 1, 1, 1],
        #[0, 1, 1, 1, 1]]���������
    #���������������0������������������������������1���������
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
        pad_var_list = torch.tensor(_pad_var_list[0]).transpose(0, 1)
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
    def __init__(self, target, references, summary, candidate_summary, rank_index, pred_rank, cand_num, sent_max_len, doc_max_len, docset_max_len, wordvocab, is_training):
        #data format is as follows:
        # text: [[],[],[]] list(list(string)) for multi-document; one per article sentence. each token is separated by a single space
        # entities: {"0":[],"1":[]...}, a dict correponding to all the sentences, one list per sentence
        # relations: list(list()) the inner list correspongding to a 3-element tuple, [ent1, relation, ent2]
        # types: list  one type per entity
        # clusters: list(list) the inner list is the set of all the co-reference lists

        # filterwords are only used in graph building process
        # all the text words should be in the range of word_vocab, or it will be [UNK]

        self.wordvocab = wordvocab
        start_decoding = wordvocab.word2id(vocabulary.START_DECODING)
        stop_decoding = wordvocab.word2id(vocabulary.STOP_DECODING)

        self.sent_max_len = sent_max_len
        self.doc_max_len = doc_max_len
        self.docset_max_len = docset_max_len

        self.raw_cand = []
        self.enc_cand = []
        self.enc_cand_nostop = []

        for index in range(cand_num):
            if is_training:
                summ = candidate_summary[rank_index[index]]
            else:
                # sel_id = rank_index[pred_rank[index]]
                # summ = candidate_summary[sel_id]
                summ = candidate_summary[rank_index[index]]
            self.raw_cand.append([])
            self.enc_cand.append([])
            self.enc_cand_nostop.append([])
            summ = nltk.sent_tokenize(summ)
            for sent in summ:
                article_words = sent.lower().split()
                if len(article_words) > sent_max_len:
                    article_words = article_words[:sent_max_len]
                self.raw_cand[index].append(article_words)
                self.enc_cand[index].append([wordvocab.word2id(w) for w in article_words])
                self.enc_cand_nostop[index].append([wordvocab.word2id(w) for w in article_words if w not in FILTERWORD])

        self.tarpaper = nltk.sent_tokenize(target)
        self.refpaper = []
        for key in references:
            ref = references[key]['abstract']
            if ref != "":
                self.refpaper.append(nltk.sent_tokenize(ref))
                if len(self.refpaper) >= self.docset_max_len:
                    break

        self.summary = summary.lower()
        self.summary = re.sub(r'@cite\_\d{1,3}','@cite',self.summary)
        summary_sents = nltk.sent_tokenize(self.summary)

        abstract_words = []
        self.abs_ids_list = []
        self.abs_ids_list_nostop = []
        for sent in summary_sents:
            article_words = sent.lower().split()
            if len(article_words) > sent_max_len:
                article_words = article_words[:sent_max_len]
            abstract_words.append(article_words)
            self.abs_ids_list.append([wordvocab.word2id(w) for w in article_words])
            self.abs_ids_list_nostop.append([wordvocab.word2id(w) for w in article_words if w not in FILTERWORD])

        abstract_words = sum(abstract_words,[])
        abs_ids = sum(self.abs_ids_list,[])
        abs_ids_no = sum(self.abs_ids_list_nostop,[])
        set_abs_ids = set(abs_ids_no)

        self.bows = []
        self.diffbows = []

        for nostop in self.enc_cand_nostop:
            set_cand_sumids = set(sum(nostop,[]))

            inter_words = set_cand_sumids.intersection(set_abs_ids)
            bows = self.generate_bow(inter_words)
            self.bows.append(bows)

            diff_words = set_cand_sumids.difference(set_abs_ids)
            diffbows = self.generate_bow(diff_words)
            self.diffbows.append(diffbows)

        # process target paper
        self.enc_input = []
        self.raw_input = []
        self.article_len = []

        self.raw_tarpaper_input = []
        
        self.article_len.append(len(self.tarpaper))
        for sent in self.tarpaper:
            article_words = sent.lower().split()
            if len(article_words) > sent_max_len:
                article_words = article_words[:sent_max_len]
            self.raw_tarpaper_input.append(article_words)
            self.raw_input.append(article_words)
            self.enc_input.append([wordvocab.word2id(w) for w in article_words])
            
            if len(self.raw_tarpaper_input) >= self.doc_max_len:
                break

        # process reference papers
        self.raw_refpaper_input = []
        for index,doc in enumerate(self.refpaper):
            docLen = len(doc)
            self.article_len.append(docLen)
            for sent in doc:
                article_words = sent.lower().split()
                if len(article_words) > sent_max_len:
                    article_words = article_words[:sent_max_len]
                self.raw_refpaper_input.append(article_words)
                self.raw_input.append(article_words)
                self.enc_input.append([wordvocab.word2id(w) for w in article_words])
                if len(self.raw_refpaper_input) >= self.doc_max_len:
                    break

        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, 200, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)
        
        self.article_oovs = []
        
        self.enc_input_extend = []
        self.enc_cand_extend = []
        self.article_oovs = []
        for enc_sent in self.raw_input:
            enc_input_extend_vocab, self.article_oovs = data.article2ids(enc_sent, wordvocab, self.article_oovs)
            #self.article_oovs.extend(oovs)
            self.enc_input_extend.append(enc_input_extend_vocab)
        for index,enc_doc in enumerate(self.raw_cand):
            self.enc_cand_extend.append([])
            for enc_sent in enc_doc:
                enc_input_extend_vocab, self.article_oovs = data.article2ids(enc_sent, wordvocab, self.article_oovs)
                #self.article_oovs.extend(oovs)
                self.enc_cand_extend[index].append(enc_input_extend_vocab)
        
        # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
        abs_ids_extend_vocab = data.abstract2ids(abstract_words, wordvocab, self.article_oovs)

        # Overwrite decoder target sequence so it uses the temp article OOV ids
        _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, 200, start_decoding, stop_decoding)
    

    def __str__(self):
        return '\n'.join([str(k)+':\t'+str(v) for k, v in self.__dict__.items()])

    def __len__(self):
        return len(self.raw_text)
    
    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len: # truncate
            inp = inp[:max_len]
            target = target[:max_len] # no end_token
        else: # no truncation
            target.append(stop_id) # end token
        assert len(inp) == len(target)
        return inp, target

    def generate_bow(self, bow):
        counter = Counter(bow)
        all_sum = sum(counter.values())
        bow = torch.zeros(self.wordvocab.size())
        if all_sum == 0:
            return bow
        keys, values = list(counter.keys()), list(counter.values()) 
        bow[keys] = torch.tensor(values).float()
        bow = bow/all_sum
        bow = bow.gt(0).float()
        bow[0] = 0
        #return torch.stack(bows,dim=0)
        return bow

class ExampleSet(torch.utils.data.Dataset):
    def __init__(self, text_path, predrank_path, rank_type, wordvocab, sent_max_len, doc_max_len, docset_max_len, cand_num, is_training = True, device=None):
        super(ExampleSet, self).__init__()
        self.device = device
        self.wordvocab = wordvocab

        self.sent_max_len = sent_max_len
        self.doc_max_len = doc_max_len
        self.docset_max_len = docset_max_len
        self.cand_num = cand_num
        self.is_training = is_training

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.json_text_list = readJson(text_path) ###���������������������������text: , summary: ,label:���
        self.pred_rank = readText(predrank_path) ###���������������������������text: , summary: ,label:���
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.json_text_list))
        self.size = len(self.json_text_list) ###������������������
        self.rank_type = rank_type

    def get_example(self, index):
        json_text = self.json_text_list[index]
        pred_rank = self.pred_rank[index%5093]
        if self.rank_type == 'RougeL':
            rank_index = json_text['rank_indexL']
        elif self.rank_type == 'Rouge1':
            rank_index = json_text['rank_index1']
        elif self.rank_type == 'Rouge2':
            rank_index = json_text['rank_index2']
        elif self.rank_type == 'Piratio':
            rank_index = json_text['rank_piratio']
        #e["summary"] = e.setdefault("summary", [])
        example = Example(json_text['target_paper'], json_text['reference'], json_text['summary'], json_text['candidate_summaries'],rank_index, pred_rank, self.cand_num, self.sent_max_len, self.doc_max_len, self.docset_max_len, self.wordvocab, self.is_training)
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
        #enc_sent_input_pad���������������������pad������������������������������������������������������������������max���������
        ex_data = self.get_tensor(item)
        return ex_data

    def __len__(self):
        return self.size

    def get_tensor(self, ex):
        
        _cached_tensor = {'summary':ex.summary, 'tgt':torch.LongTensor(ex.dec_input), \
                            'example':ex, 'tgt_extend': torch.LongTensor(ex.target), 'oovs':ex.article_oovs, 'article_len':ex.article_len, \
                            'text': [torch.LongTensor(x) for x in ex.enc_input], 'text_extend': [torch.LongTensor(x) for x in ex.enc_input_extend],\
                            'cand_summary': [[torch.LongTensor(x) for x in y] for y in ex.enc_cand], 'cand_summary_extend': [[torch.LongTensor(x) for x in y] for y in ex.enc_cand_extend], 'bows':ex.bows, 'diffbows':ex.diffbows, 'tgt_enc': [torch.LongTensor(x) for x in ex.abs_ids_list]}
        return _cached_tensor

    def batch_fn(self, samples):
        batch_examples, batch_tgt, batch_tgt_extend,batch_raw_tgt_text,batch_oovs, \
            batch_art_len, batch_text,batch_text_extend, batch_cand,batch_cand_extend, batch_bows, batch_tgt_enc, batch_diffbows  =  [], [], [], [], [], [], [], [], [], [],[],[],[]
        #batch_ex = map(list, zip(*samples))
        for ex_data in samples:
            if ex_data != {}:
                batch_examples.append(ex_data['example'])
                batch_tgt.append(ex_data['tgt'])
                batch_raw_tgt_text.append(ex_data['summary'])
                batch_oovs.append(ex_data['oovs'])
                batch_tgt_extend.append(ex_data['tgt_extend'])
                batch_art_len.append(ex_data['article_len'])
                batch_text.append(ex_data['text'])
                batch_text_extend.append(ex_data['text_extend'])
                batch_cand.extend(ex_data['cand_summary'])
                batch_cand_extend.extend(ex_data['cand_summary_extend'])
                batch_bows.extend(ex_data['bows'])
                batch_diffbows.extend(ex_data['diffbows'])
                batch_tgt_enc.append(ex_data['tgt_enc'])


        pad_id = self.wordvocab.word2id('<PAD>')
        bos_id = self.wordvocab.word2id('<BOS>')
        eos_id = self.wordvocab.word2id('<EOS>')
        
        batch_tgt = pad_sent_entity(batch_tgt, pad_id,bos_id,eos_id, flatten = False)
        batch_tgt_extend = pad_sent_entity(batch_tgt_extend, pad_id,bos_id,eos_id, flatten = False)
        batch_text,sent_num = pad_sent_entity(batch_text, pad_id,bos_id,eos_id, flatten = True)
        batch_text_extend,sent_num = pad_sent_entity(batch_text_extend, pad_id,bos_id,eos_id, flatten = True)

        batch_cand,_ = pad_sent_entity(batch_cand, pad_id,bos_id,eos_id, flatten = True)
        batch_cand_extend, _ = pad_sent_entity(batch_cand, pad_id,bos_id,eos_id, flatten = True)


        batch_bows = torch.stack(batch_bows,dim=0)
        batch_diffbows = torch.stack(batch_diffbows,dim=0)
        batch_tgt_enc, _ = pad_sent_entity(batch_tgt_enc, pad_id, bos_id,eos_id, flatten = True)

        max_art_oovs = max([len(oovs) for oovs in batch_oovs])
        extra_zeros = None
        batch_size = batch_tgt.shape[1]
        if max_art_oovs > 0:
            #extra_zeros = Variable(torch.zeros((batch_size, max_art_oovs)))
            extra_zeros = torch.zeros((batch_size, max_art_oovs))
        return {'extra_zeros':extra_zeros, 'examples':batch_examples, 'tgt': batch_tgt, 'sent_num':sent_num, 'raw_tgt_text': batch_raw_tgt_text, \
            'article_oovs':batch_oovs, 'tgt_extend': batch_tgt_extend, 'article_len':batch_art_len,\
            'text':batch_text,'text_extend':batch_text_extend,'cand':batch_cand,'cand_extend':batch_cand_extend,\
            'bows':batch_bows, 'diffbows':batch_diffbows, 'tgt_enc':batch_tgt_enc}
        
if __name__ == '__main__' :
    pass

