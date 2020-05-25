### TAKEN FROM https://github.com/kolloldas/torchnlp

import math
# from numpy import random
import os
import pprint
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from model.common_layer import (DecoderLayer, EncoderLayer, LabelSmoothing,
                                LayerNorm, MultiHeadAttention, NoamOpt,
                                PositionwiseFeedForward, SoftmaxOutputLayer,
                                VarDecoderLayer1, _gen_bias_mask,
                                _gen_timing_signal, _get_attn_self_mask,
                                _get_attn_subsequent_mask, gaussian_kld,
                                get_input_from_batch, get_output_from_batch,
                                share_embedding, pad)
from utils import config

pp = pprint.PrettyPrinter(indent=1)


def init_rnn_wt(rnn):
    for names in rnn._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(rnn, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)


def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)


class WordEncoder(nn.Module):
    """
    A Word Level Encoder
    """
    def __init__(self, embedding_size, hidden_size, bidirectional):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
        """
        super(WordEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional)
        # init_rnn_wt(self.rnn)
        

    def forward(self, inputs, input_length):
        # sort first
        x_len_sorted, indices = input_length.sort(descending=True)
        x = inputs.index_select(0, indices)
        
        # rnn encode
        packed = pack_padded_sequence(x, x_len_sorted, batch_first=True)
        output, hidden = self.rnn(packed)
        y, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        y = y.contiguous()
        
        # reorder
        _, inverse_indices = indices.sort()
        y = y.index_select(0, inverse_indices)
        hidden = hidden.index_select(1, inverse_indices)
        return y, hidden


class Encoder(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0, layer_dropout=0, 
                 attention_dropout=0.1, relu_dropout=0.1, use_mask=False, universal=False):
        """
        Parameters:
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        
        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        if(self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])
        
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        
        if(config.act):
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, mask):
        #Add input dropout
        x = self.input_dropout(inputs)
        
        if(self.universal):
            if(config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            
            for i in range(self.num_layers):
                x = self.enc[i](x, mask)
        
            y = self.layer_norm(x)
        return y

class VarDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, vocab_size, max_length=200, input_dropout=0, layer_dropout=0, 
                 attention_dropout=0.1, relu_dropout=0.1, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(VarDecoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length), # mandatory
                 vocab_size,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        self.vardec = nn.Sequential(*[VarDecoderLayer1(*params) for l in range(config.num_var_layers)])
        self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers- config.num_var_layers)])
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm_1 = LayerNorm(hidden_size)
        self.layer_norm_2 = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, posterior, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg.byte() + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        dec_mask_p = mask_trg
        #Add input dropout

        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
            
        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # Run decoder
        y, _, _, attn_dist, _ , means, log_vars, logits_probs= self.vardec((x, encoder_output, posterior, [], (mask_src,dec_mask,dec_mask_p),{"prior":[],"posterior":[]},{"prior":[],"posterior":[]},[]))
        y = self.layer_norm_1(y)
        y, _, attn_dist, _ = self.dec((y, encoder_output, [], (mask_src,dec_mask)))
        y = self.layer_norm_2(y)
        if posterior:
            #print(means["prior"].shape)
            means["prior"] = torch.cat(means["prior"], 0)
            means["posterior"] = torch.cat(means["posterior"], 0)
            log_vars["prior"] = torch.cat(log_vars["prior"], 0)
            log_vars["posterior"] = torch.cat(log_vars["posterior"], 0)
            
        return y, attn_dist, means, log_vars, logits_probs

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, beam_search=False, attn_dist_db=None):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if(config.pointer_gen):
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist/temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist            
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)]*x.size(1),1) ## extend for all seq
            if(beam_search):
                enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab_[0].unsqueeze(0)]*x.size(0),0) ## extend for all seq
            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_))
            return logit
        else:
            return F.log_softmax(logit,dim=-1)


class CvaeTrans(nn.Module):

    def __init__(self, vocab, emo_number,  model_file_path=None, is_eval=False, load_optim=False):
        super(CvaeTrans, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, pretrain=False)
        self.word_encoder = WordEncoder(config.emb_dim, config.hidden_dim, config.bidirectional)
        self.linear = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        self.encoder = Encoder(config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter,universal=config.universal)
        
        self.decoder = VarDecoder(config.emb_dim, hidden_size = config.hidden_dim,  num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth,total_value_depth=config.depth,
                                filter_size=config.filter, vocab_size = self.vocab_size)

        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        
        # if model_file_path:
        #     print("loading weights")
        #     state = torch.load(model_file_path, map_location= lambda storage, location: storage)
        #     self.encoder.load_state_dict(state['encoder_state_dict'])
        #     #self.r_encoder.load_state_dict(state['r_encoder_state_dict'])
        #     self.decoder.load_state_dict(state['decoder_state_dict'])
        #     self.generator.load_state_dict(state['generator_dict'])
        #     self.embedding.load_state_dict(state['embedding_dict'])
            
        if (config.USE_CUDA):
            self.cuda()
        if is_eval:
            self.eval()
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
            if(config.noam):
                self.optimizer = NoamOpt(config.hidden_dim, 1, 8000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
            if (load_optim):
                self.optimizer.load_state_dict(state['optimizer'])
                if config.USE_CUDA:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b):

        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            #'r_encoder_state_dict': self.r_encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, iter, train=True):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        if(train):
            if(config.noam):
                self.optimizer.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()

        ## Encode
        num_sentences, enc_seq_len = enc_batch.size()
        batch_size = enc_lens.size(0)
        max_len = enc_lens.data.max().item()
        input_lengths = torch.sum(~enc_batch.data.eq(config.PAD_idx), dim=1)
        
        # word level encoder
        enc_emb = self.embedding(enc_batch)
        word_encoder_outpus, word_encoder_hidden = self.word_encoder(enc_emb, input_lengths)
        word_encoder_hidden = word_encoder_hidden.transpose(1, 0).reshape(num_sentences, -1)

        # pad and pack word_encoder_hidden
        start = torch.cumsum(torch.cat((enc_lens.data.new(1).zero_(), enc_lens[:-1])), 0)
        word_encoder_hidden = torch.stack([pad(word_encoder_hidden.narrow(0, s, l), max_len)
                                            for s, l in zip(start.data.tolist(), enc_lens.data.tolist())], 0)
        mask_src = ~(enc_padding_mask.bool()).unsqueeze(1)

        # context level encoder
        if word_encoder_hidden.size(-1) != config.hidden_dim:
            word_encoder_hidden = self.linear(word_encoder_hidden)
        encoder_outputs = self.encoder(word_encoder_hidden, mask_src)

        # Decode
        sos_token = torch.LongTensor([config.SOS_idx] * batch_size).unsqueeze(1)
        if config.USE_CUDA: sos_token = sos_token.cuda()
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]),1)

        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        pre_logit, attn_dist, mean, log_var, probs= self.decoder(self.embedding(dec_batch_shift), encoder_outputs, True, (mask_src, mask_trg))
        ## compute output dist
        logit = self.generator(pre_logit, attn_dist, enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros, attn_dist_db=None)
        ## loss: NNL if ptr else Cross entropy
        sbow = dec_batch #[batch, seq_len]
        seq_len = sbow.size(1)
        loss_rec = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        if config.model=="cvaetrs":
            loss_aux = 0
            for prob in probs:
                sbow_mask = _get_attn_subsequent_mask(seq_len).transpose(1,2)
                sbow.unsqueeze(2).repeat(1,1,seq_len).masked_fill_(sbow_mask,config.PAD_idx)#[batch, seq_len, seq_len]

                loss_aux+= self.criterion(prob.contiguous().view(-1, prob.size(-1)), sbow.contiguous().view(-1))
            kld_loss = gaussian_kld(mean["posterior"], log_var["posterior"],mean["prior"], log_var["prior"])
            kld_loss = torch.mean(kld_loss)
            kl_weight = min(math.tanh(6 * iter/config.full_kl_step - 3) + 1, 1)
            #kl_weight = min(iter/config.full_kl_step, 1) if config.full_kl_step >0 else 1.0
            loss = loss_rec + config.kl_ceiling * kl_weight*kld_loss + config.aux_ceiling*loss_aux
            elbo = loss_rec+kld_loss
        else:
            loss = loss_rec
            elbo = loss_rec
            kld_loss = torch.Tensor([0])
            loss_aux = torch.Tensor([0])
        if(train):
            loss.backward()
            # clip gradient
            nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
            self.optimizer.step()

        return loss_rec.item(), math.exp(min(loss_rec.item(), 100)), kld_loss.item(), loss_aux.item(), elbo.item()

    def train_n_batch(self, batchs, iter, train=True):
        if(config.noam):
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
        for batch in batchs:
            enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
            dec_batch, _, _, _, _ = get_output_from_batch(batch)
            ## Encode
            mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
            encoder_outputs = self.encoder(self.embedding(enc_batch), mask_src)

            meta = self.embedding(batch["program_label"])
            if config.dataset=="empathetic":
                meta = meta-meta
            # Decode
            sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
            if config.USE_CUDA: sos_token = sos_token.cuda()
            dec_batch_shift = torch.cat((sos_token,dec_batch[:, :-1]),1)

            mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

            pre_logit, attn_dist, mean, log_var, probs= self.decoder(self.embedding(dec_batch_shift)+meta.unsqueeze(1),encoder_outputs, True, (mask_src,mask_trg))
            ## compute output dist
            logit = self.generator(pre_logit,attn_dist,enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros, attn_dist_db=None)
            ## loss: NNL if ptr else Cross entropy
            sbow = dec_batch #[batch, seq_len]
            seq_len = sbow.size(1)
            loss_rec = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
            if config.model=="cvaetrs":
                loss_aux = 0
                for prob in probs:
                    sbow_mask = _get_attn_subsequent_mask(seq_len).transpose(1,2)
                    sbow.unsqueeze(2).repeat(1,1,seq_len).masked_fill_(sbow_mask,config.PAD_idx)#[batch, seq_len, seq_len]

                    loss_aux+= self.criterion(prob.contiguous().view(-1, prob.size(-1)), sbow.contiguous().view(-1))
                kld_loss = gaussian_kld(mean["posterior"], log_var["posterior"],mean["prior"], log_var["prior"])
                kld_loss = torch.mean(kld_loss)
                kl_weight = min(math.tanh(6 * iter/config.full_kl_step - 3) + 1, 1)
                #kl_weight = min(iter/config.full_kl_step, 1) if config.full_kl_step >0 else 1.0
                loss = loss_rec + config.kl_ceiling * kl_weight*kld_loss + config.aux_ceiling*loss_aux
                elbo = loss_rec+kld_loss
            else:
                loss = loss_rec
                elbo = loss_rec
                kld_loss = torch.Tensor([0])
                loss_aux = torch.Tensor([0])
            loss.backward()
            # clip gradient
        nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
        self.optimizer.step()

        return loss_rec.item(), math.exp(min(loss_rec.item(), 100)), kld_loss.item(), loss_aux.item(), elbo.item()

    def decoder_greedy(self, batch, max_dec_step=50):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        
        ## Encode
        num_sentences, enc_seq_len = enc_batch.size()
        batch_size = enc_lens.size(0)
        max_len = enc_lens.data.max().item()
        input_lengths = torch.sum(~enc_batch.data.eq(config.PAD_idx), dim=1)
        
        # word level encoder
        enc_emb = self.embedding(enc_batch)
        word_encoder_outpus, word_encoder_hidden = self.word_encoder(enc_emb, input_lengths)
        word_encoder_hidden = word_encoder_hidden.transpose(1, 0).reshape(num_sentences, -1)

        # pad and pack word_encoder_hidden
        start = torch.cumsum(torch.cat((enc_lens.data.new(1).zero_(), enc_lens[:-1])), 0)
        word_encoder_hidden = torch.stack([pad(word_encoder_hidden.narrow(0, s, l), max_len)
                                            for s, l in zip(start.data.tolist(), enc_lens.data.tolist())], 0)
        mask_src = ~(enc_padding_mask.bool()).unsqueeze(1)

        # context level encoder
        if word_encoder_hidden.size(-1) != config.hidden_dim:
            word_encoder_hidden = self.linear(word_encoder_hidden)
        encoder_outputs = self.encoder(word_encoder_hidden, mask_src)
        
        ys = torch.ones(batch_size, 1).fill_(config.SOS_idx).long()
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        
        decoded_words = []
        for i in range(max_dec_step+1):

            out, attn_dist, _, _,_ = self.decoder(self.embedding(ys), encoder_outputs, False, (mask_src, mask_trg))
            
            prob = self.generator(out,attn_dist,enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
            _, next_word = torch.max(prob[:, -1], dim = 1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in next_word.view(-1)])

            if config.USE_CUDA:
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
                ys = ys.cuda()
            else:
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
            
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+= e + ' '
            sent.append(st)
        return sent
