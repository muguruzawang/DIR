from module.transformer_encoder import NewTransformerEncoder, BertLSTMEncoder
from module.neural import PositionwiseFeedForward, sequence_mask
from module.roberta import RobertaEmbedding
from modules import GraphTrans

import torch.nn as nn
import torch
import pdb

from module.utlis_dataloader import *

class EMEncoder(nn.Module):
    def __init__(self, args, device, src_embeddings, padding_idx, bert_model):
        super(EMEncoder, self).__init__()
        self.args = args
        self.padding_idx = padding_idx
        self.device = device
        # self._TFembed = nn.Embedding(50, self.args.emb_size) # box=10 , embed_size = 256

        if args.use_bert:
            self.bert_model = bert_model
            self.sent_encoder = BertLSTMEncoder(self.bert_model)
            self.entity_encoder = BertLSTMEncoder(self.bert_model)
        else:
            self.sent_encoder = NewTransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                      self.args.ff_size, self.args.enc_dropout, src_embeddings, self.device)
            self.entity_encoder = NewTransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                      self.args.ff_size, self.args.enc_dropout, src_embeddings, self.device)

        self.graph_enc = GraphTrans(args)

        self.layer_norm = nn.LayerNorm(self.args.emb_size, eps=1e-6)
        # self.layer_norm2 = nn.LayerNorm(self.args.emb_size, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(self.args.enc_hidden_size, self.args.ff_size, self.args.enc_dropout)



    def forward(self, batch, topic_info, article_len):
        """
        :param src:  batch_size x n_paras x n_tokens
        :param cluster: batch_size x n_clusters x n_cluster_tokens
        :param edge: batch_size x n_paras x n_clusters
        :return:
        """
        src = batch['text']
        #print(src.size())

        batch_size, n_sents, n_tokens = src.size()
        #print(ent.size())
        sent_feature, sent_context, _ = self.sent_encoder(src, topic_info, article_len) 
        
        return sent_feature, sent_context
        
        #return sent_feature, sent_context, _,_