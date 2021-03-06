"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
from util import masked_softmax

class BERT(nn.Module):
    """BERT model for SQuAD.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BERT, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=300,
                                    drop_prob=drop_prob)


        self.bert_start = nn.Linear(
            in_features = 300,
            out_features = 1,
            bias = True
        )
        nn.init.xavier_uniform_(self.bert_start.weight,gain=1)

        self.bert_end = nn.Linear(
            in_features = 300,
            out_features = 1,
            bias = True
        )
        nn.init.xavier_uniform_(self.bert_end.weight,gain=1)

        self.proj_up = nn.Linear(
            in_features = 300,
            out_features = hidden_size,
            bias = True
        )

        self.proj_down = nn.Linear(
            in_features = hidden_size,
            out_features = 300,
            bias = True
        )
        nn.init.xavier_uniform_(self.proj_down.weight,gain=1)



    def forward(self, cw_idxs, qw_idxs, bert_embeddings, max_context_len, max_question_len, device):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        #c_len, q_len = max_context_len, max_question_len
        glove_c_emb = self.emb(cw_idxs) # (batch_size, c_len, 300)
        glove_q_emb = self.emb(qw_idxs) # (batch_size, c_len, 300)
        #print("word_vec_emb.size() before : ", word_vec_emb.size())
        #word_vec_emb = self.proj_up(word_vec_emb) # (batch_size, c_len, hidden_size)
        #print("word_vec_emb.size() after: ", word_vec_emb.size())

        c_emb = bert_embeddings[:,0:torch.max(c_len),:] # (batch_size, c_len, hidden_size)
        c_emb = self.proj_down(torch.nn.functional.relu(c_emb)) # (batch_size, c_len, 300)
        c_emb = c_emb + c_emb * glove_c_emb

        start_logits = self.bert_start(c_emb) # (batch_size, c_len, 1)
        end_logits = self.bert_end(c_emb) # (batch_size, c_len, 1)

        log_p1 = masked_softmax(start_logits.squeeze(), c_mask, log_softmax=True) # (batch_size, c_len)
        log_p2 = masked_softmax(end_logits.squeeze(), c_mask, log_softmax=True) # (batch_size, c_len)

        out = log_p1, log_p2
        return out # 2 tensors, each (batch_size, c_len)


class DCN(nn.Module):
    """ Dynamic Coattention Network

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, max_context_len, max_question_len, drop_prob=0.):
        super(DCN, self).__init__()
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.DCNAttention(hidden_size=2 * hidden_size,
                                         max_context_len = max_context_len,
                                         max_question_len = max_question_len,
                                         drop_prob = drop_prob
                                         )

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

        self.proj_bert_down = nn.Linear(
            in_features = 768,
            out_features = hidden_size,
            bias = True
        )
        nn.init.xavier_uniform_(self.proj_bert_down.weight,gain=1)

        self.proj_glove_down = nn.Linear(
            in_features = 300,
            out_features = hidden_size,
            bias = True
        )
        nn.init.xavier_uniform_(self.proj_glove_down.weight,gain=1)

    def forward(self, cw_idxs, qw_idxs, bert_embeddings, max_context_len, max_question_len, device):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        ## Addition to enforce max_len for context and question
        c_mask = c_mask[:,:max_context_len]
        q_mask = q_mask[:,:max_question_len]
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        if bert_embeddings is not None:
            bert_embeddings = torch.nn.functional.relu(self.proj_bert_down(bert_embeddings))
            bert_c_emb = bert_embeddings[:,0:torch.max(c_len),:]
            bert_q_emb = bert_embeddings[:,max_context_len: max_context_len + torch.max(q_len),:]

            c_emb = self.emb(cw_idxs)[:,:max_context_len,:]         # (batch_size, c_len, 300)
            q_emb = self.emb(qw_idxs)[:,:max_question_len,:]         # (batch_size, q_len, 300)
            if(c_emb.size(1) > max_context_len or q_emb.size(1) > max_question_len):
                print("c_emb.size() ", c_emb.size())
                print("q_emb.size() ", q_emb.size())

            c_emb = c_emb  +  c_emb * bert_c_emb # (batch_size, c_len, 100/ hidden_size)
            q_emb = q_emb + q_emb * bert_q_emb # (batch_size, q_len, 100/ hidden_size)
        else:
            c_emb = self.emb(cw_idxs)[:,:max_context_len,:]         # (batch_size, c_len, 300)
            q_emb = self.emb(qw_idxs)[:,:max_question_len,:]         # (batch_size, q_len, 300)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask, device)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

        self.proj_bert_down = nn.Linear(
            in_features = 768,
            out_features = hidden_size,
            bias = True
        )
        nn.init.xavier_uniform_(self.proj_bert_down.weight,gain=1)

        self.proj_glove_down = nn.Linear(
            in_features = 300,
            out_features = hidden_size,
            bias = True
        )
        nn.init.xavier_uniform_(self.proj_glove_down.weight,gain=1)

    def forward(self, cw_idxs, qw_idxs, bert_embeddings, max_context_len, max_question_len, device):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        ## Addition to enforce max_len for context and question
        c_mask = c_mask[:,:max_context_len]
        q_mask = q_mask[:,:max_question_len]
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        bert_embeddings = torch.nn.functional.relu(self.proj_bert_down(bert_embeddings))
        bert_c_emb = bert_embeddings[:,0:torch.max(c_len),:]
        bert_q_emb = bert_embeddings[:,max_context_len: max_context_len + torch.max(q_len),:]

        c_emb = self.emb(cw_idxs)[:,:max_context_len,:]         # (batch_size, c_len, 300)
        q_emb = self.emb(qw_idxs)[:,:max_question_len,:]         # (batch_size, q_len, 300)
        if(c_emb.size(1) > max_context_len or q_emb.size(1) > max_question_len):
            print("c_emb.size() ", c_emb.size())
            print("q_emb.size() ", q_emb.size())

        c_emb = c_emb  +  c_emb * bert_c_emb # (batch_size, c_len, 100/ hidden_size)
        q_emb = q_emb + q_emb * bert_q_emb # (batch_size, q_len, 100/ hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
