import torch
from modules import BiLSTM, GraphTrans, GAT_Hetersum, MSA 
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.nn.init import xavier_uniform_

import dgl
from module.embedding import Word_Embedding
from module.transformer_decoder import TransformerDecoder
from module.EMNewEncoder import EMEncoder
from module.optimizer import Optimizer
from module.neural import PositionwiseFeedForward2
import copy
import pdb
import numpy as np

def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Longformer(nn.Module):
    def __init__(self, path, vocab_size):
        super(Longformer, self).__init__()
        self.model = AutoModel.from_pretrained(path)

    def forward(self, x, attention_mask):
        top_vec = self.model(x, attention_mask=attention_mask)
        return top_vec

class ERTSumGraph(nn.Module):
    def __init__(self, args, padding_idx, vocab_size, device, checkpoint=None):
        super(ERTSumGraph, self).__init__()
        self.args = args
        self.device = device
        self.padding_idx = padding_idx
    
        self.vocab_size = vocab_size        
        self.longformer = Longformer(r'/data/run01/scv6131/wpc/Prompt_Contrastive_BCELoss/longformer-base-4096', self.vocab_size)

        self.ffd = PositionwiseFeedForward2(768*2, 2048, 1, 0.1)

        self.sigmoid = nn.Sigmoid()

        if checkpoint is not None:
            keys = list(checkpoint['model'].keys())
            for k in keys:
                if ('a_2' in k):
                    checkpoint['model'][k.replace('a_2', 'weight')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])
                if ('b_2' in k):
                    checkpoint['model'][k.replace('b_2', 'bias')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for n, p in self.named_parameters():
                if 'longformer' not in n:
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self,  batch):
        inputs = batch['input']
        summaries = batch['summaries']

        attention_mask = inputs.ne(self.padding_idx)
        global_attention_mask = inputs.eq(0)

        outputs = self.longformer(inputs, attention_mask)

        outputs = outputs.last_hidden_state

        source_state = outputs[:,0:1,:]


        summaries = summaries.view(-1,summaries.size(-1))
        sum_mask = summaries.ne(self.padding_idx)

        global_attention_mask = summaries.eq(0)

        sum_outputs = self.longformer(summaries, sum_mask)
        sum_outputs = sum_outputs.last_hidden_state

        summary_state = sum_outputs[:,0,:]
        summary_state = summary_state.view(inputs.size(0),-1, summary_state.size(-1))

        source_state = source_state.expand(summary_state.shape)

        summary_state = torch.cat((source_state,summary_state),dim=-1)

        pred_scores = self.sigmoid(self.ffd(summary_state))

        scores = {}

        scores['rank_scores'] = pred_scores

        pos_idx = np.random.choice(2)

        neg_idx = np.random.choice(2)

        pos_scores = pred_scores[:,pos_idx,]
        neg_scores = pred_scores[:,4+neg_idx,]

        bce_scores = torch.cat((pos_scores,neg_scores),dim=0)

        scores['bce_scores'] = bce_scores
        
        return scores