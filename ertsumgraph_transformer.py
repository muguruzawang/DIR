import torch
from modules import BiLSTM, GraphTrans, GAT_Hetersum, MSA 
from torch import nn
from transformers import RobertaTokenizer, RobertaModel
from torch.nn.init import xavier_uniform_

import dgl
from module.embedding import Word_Embedding
from module.EMDecoder import TransformerDecoder as Hier_TransformerDecoder
from module.neural import PositionwiseFeedForward2
from module.transformer_decoder import TransformerDecoder as Sin_TransformerDecoder
from module.roberta import RobertaEmbedding
from module.EMNewEncoder import EMEncoder
from module.optimizer import Optimizer
import pdb

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps_bert,
            model_size=args.enc_hidden_size)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert_model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps_dec,
            model_size=args.enc_hidden_size)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert_model')]
    optim.set_parameters(params)


    return optim

def get_generator(dec_hidden_size, vocab_size, emb_dim, device):
    gen_func = nn.Softmax(dim=-1)
    ### nn.Sequential内部实现了forward函数
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, emb_dim),
        nn.LeakyReLU(),
        nn.Linear(emb_dim, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Roberta_model(nn.Module):
    def __init__(self, roberta_path, finetune=False):
        super(Roberta_model, self).__init__()
        print('Roberta initialized')
        self.model = RobertaModel.from_pretrained(roberta_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
        self.pad_id = self.tokenizer.pad_token_id
        self._embedding = self.model.embeddings.word_embeddings

        self.finetune = finetune

    def forward(self, input_ids):
        attention_mask = (input_ids != self.pad_id).float()
        if(self.finetune):
            return self.model(input_ids, attention_mask=attention_mask)
        else:
            self.eval()
            with torch.no_grad():
                return self.model(input_ids, attention_mask=attention_mask)

class ERTSumGraph(nn.Module):
    def __init__(self, args, word_padding_idx, vocab_size, device, checkpoint=None):
        super(ERTSumGraph, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.padding_idx = word_padding_idx
        self.device = device
        # need to encode the following nodes:
        # word embedding : use glove embedding
        # sentence encoder: bilstm (word)
        # doc encoder: bilstm (sentence)
        # entity encoder: bilstm (word)
        # relation embedding: initial embedding
        # type embedding: initial embedding 

        # use roberta
        
        self.src_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)
        tgt_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)
        if self.args.share_embeddings:
            tgt_embeddings.weight = self.src_embeddings.weight
        self.encoder = EMEncoder(self.args, self.device, self.src_embeddings, self.padding_idx, None)
        emb_dim = tgt_embeddings.weight.size(1)
        
        self.generator = get_generator(self.args.dec_hidden_size, self.vocab_size, emb_dim, self.device)
        if self.args.share_decoder_embeddings:
            self.generator[2].weight = tgt_embeddings.weight

        if args.hier_decoder:
            self.decoder = Hier_TransformerDecoder(
                self.args.dec_layers,
                self.args.dec_hidden_size, heads=self.args.heads,
                d_ff=self.args.ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
        
        else:
            self.decoder = Sin_TransformerDecoder(
                self.args.dec_layers, self.args.cand_num, self.args.use_z2,
                self.args.dec_hidden_size, heads=self.args.heads,
                d_ff=self.args.ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings, generator=self.generator)

        self.mean = PositionwiseFeedForward2(self.args.emb_size*2, self.args.ff_size, self.args.emb_size, self.args.dec_dropout)
        self.logvar = PositionwiseFeedForward2(self.args.emb_size*2, self.args.ff_size, self.args.emb_size, self.args.dec_dropout)
        self.post_mean = PositionwiseFeedForward2(self.args.emb_size*3, self.args.ff_size, self.args.emb_size, self.args.dec_dropout)
        self.post_logvar = PositionwiseFeedForward2(self.args.emb_size*3, self.args.ff_size, self.args.emb_size, self.args.dec_dropout)

        self.z_supervision = nn.Sequential(nn.Linear(self.args.emb_size, vocab_size), nn.LogSoftmax(dim=-1))

        self.mean2 = PositionwiseFeedForward2(self.args.emb_size*2, self.args.ff_size, self.args.emb_size, self.args.dec_dropout)
        self.logvar2 = PositionwiseFeedForward2(self.args.emb_size*2, self.args.ff_size, self.args.emb_size, self.args.dec_dropout)
        self.post_mean2 = PositionwiseFeedForward2(self.args.emb_size*3, self.args.ff_size, self.args.emb_size, self.args.dec_dropout)
        self.post_logvar2 = PositionwiseFeedForward2(self.args.emb_size*3, self.args.ff_size, self.args.emb_size, self.args.dec_dropout)

        self.z_supervision2 = nn.Sequential(nn.Linear(self.args.emb_size, vocab_size), nn.LogSoftmax(dim=-1))

        self.z_supervision_adv = nn.Sequential(nn.Linear(self.args.emb_size, vocab_size), nn.LogSoftmax(dim=-1))
        self.z_supervision2_adv = nn.Sequential(nn.Linear(self.args.emb_size, vocab_size), nn.LogSoftmax(dim=-1))
        self.sigmoid = nn.Sigmoid()
        #self.tw_embedding = nn.Parameter(embeddings)
        if checkpoint is not None:
            # checkpoint['model']
            keys = list(checkpoint['model'].keys())
            print('keys为:'+str(keys))
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
                if 'RobertaModel' not in n:
                    if p.dim() > 1:
                        xavier_uniform_(p)


        self.to(device)

        '''
        #pdb.set_trace()
        ###查看一下roberta的模型参数是什么样的，好设置warmup_up rate
        for name, parameters in self.named_parameters():
            if name.find('bert_model')!=-1:
                print(name, ':')
        
        pdb.set_trace()

        '''
    def sample_z1(self, mu, logvar):
        epsilon = torch.empty_like(logvar).float().normal_()
        std = torch.sqrt(torch.exp(logvar))
        z = mu + std * epsilon
        return z

    def gaussian_kld(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                                   - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                                   - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)),dim=-1)
        return kld

    def forward(self,  batch):
        """
        :param src:  batch_size x n_paras x n_tokens
        :param cluster: batch_size x n_clusters x n_cluster_tokens
        :param tgt: n_tgt_tokens x batch_size
        :param edge: batch_size x n_paras x n_clusters
        :return:
        """
        tgt = batch['tgt']
        #tgt_extend = batch['tgt_extend']
        src = batch['text']
        src_extend = batch['text_extend']
        cand = batch['cand_extend']

        tgt_enc = batch['tgt_enc']
        sent_feature, sent_context, cand_feature, tgt_feature = self.encoder(batch)

        ### src state
        mask_src = src.ne(self.padding_idx)
        x,y,z = sent_feature.size()
        mask_total = torch.sum(mask_src,dim=-1)
        mask_total = mask_total.ne(0)

        src_num_sum =torch.sum(mask_total,dim=1,keepdim=True)
        src_num_sum[src_num_sum==0] = 1
        mask_total = mask_total.unsqueeze(-1).expand(x,y,z)      
        mask_src_sum = torch.sum(sent_feature*mask_total, dim=1) 
        src_state = torch.div(mask_src_sum,src_num_sum)

        ### cand state        
        # cand = cand.view(src.size(0), -1, cand.size(-2) ,cand.size(-1))
        mask_cand = cand.ne(self.padding_idx)
        x,y,z = cand_feature.size()
        mask_total = torch.sum(mask_cand,dim=-1)
        mask_total = mask_total.ne(0)

        cand_num_sum =torch.sum(mask_total,dim=1,keepdim=True)
        cand_num_sum[cand_num_sum==0] = 1
        mask_total = mask_total.unsqueeze(-1).expand(x,y,z)      
        mask_cand_sum = torch.sum(cand_feature*mask_total, dim=1) 
        cand_state = torch.div(mask_cand_sum,cand_num_sum)

        ### tgt state
        mask_tgt = tgt_enc.ne(self.padding_idx)
        x,y,z = tgt_feature.size()
        mask_total = torch.sum(mask_tgt,dim=-1)
        mask_total = mask_total.ne(0)

        tgt_num_sum =torch.sum(mask_total,dim=1,keepdim=True)
        tgt_num_sum[tgt_num_sum==0] = 1
        mask_total = mask_total.unsqueeze(-1).expand(x,y,z)      
        mask_tgt_sum = torch.sum(tgt_feature*mask_total, dim=1) 
        tgt_state = torch.div(mask_tgt_sum,tgt_num_sum)

        cand = cand.view(src.size(0), -1, cand.size(-2) ,cand.size(-1))

        if self.training:
            src_state = src_state.unsqueeze(1).repeat(1, cand.size(1), 1)
            src_state = src_state.view(-1, src_state.size(-1))
            tgt_state = tgt_state.unsqueeze(1).repeat(1, cand.size(1), 1)
            tgt_state = tgt_state.view(-1, tgt_state.size(-1))
            prior_state = torch.cat((src_state,cand_state),dim=-1)
            post_state = torch.cat((src_state,cand_state,tgt_state),dim=-1)
            mean = self.mean(prior_state)
            log_var = self.logvar(prior_state)

            post_mean = self.post_mean(post_state)
            post_log_var = self.post_logvar(post_state)
            z = self.sample_z1(post_mean, post_log_var)

            kl_loss = self.gaussian_kld(post_mean, post_log_var, mean, log_var)
            kl_loss = torch.sum(kl_loss)

            logits_prob = self.z_supervision(z)

            mean2 = self.mean2(prior_state)
            log_var2 = self.logvar2(prior_state)

            post_mean2 = self.post_mean2(post_state)
            post_log_var2 = self.post_logvar2(post_state)
            z2 = self.sample_z1(post_mean2, post_log_var2)

            kl_loss2 = self.gaussian_kld(post_mean2, post_log_var2, mean2, log_var2)
            kl_loss2 = torch.sum(kl_loss2)

            logits_prob2 = self.z_supervision2(z2)

            logits_prob_adv = self.z_supervision_adv(z2.clone().detach())
            logits_prob2_adv = self.z_supervision2_adv(z.clone().detach())
        else:
            prior_state = torch.cat((src_state,cand_state),dim=-1)
            mean = self.mean(prior_state)
            log_var = self.logvar(prior_state)

            z = self.sample_z1(mean, log_var)

        src_extend = src_extend.view(src_extend.size(0), -1)
        cand = cand.view(cand.size(0), -1)
        src_extend = torch.cat((src_extend,cand),dim=1)

        z = z.view(src.size(0),-1, z.size(-1))
        z2 = z2.view(src.size(0),-1, z2.size(-1))
        dec_state = self.decoder.init_decoder_state(src_extend, sent_context)     # src: num_paras_in_one_batch x max_length
        decoder_outputs = self.decoder(tgt,sent_context, z, z2, dec_state)
        # tgt, memory_bank, state, edge, cluster, cluster_feature

        return decoder_outputs, kl_loss, logits_prob, kl_loss2, logits_prob2, logits_prob_adv, logits_prob2_adv