"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        """
        print("orig_len: " , orig_len)
        """
        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        """
        print("x.size(): " , x.size())
        """
        x = pack_padded_sequence(x, lengths, batch_first=True)
        #print("After pack_padded_sequence - x.size(): " , x.size())

        #assert orig_len <= torch.max(lengths)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)
        #print("After rnn - x.size(): " , x.size())

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x

class DCNAttention(nn.Module):
    """Dynamic CoAttention .

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, max_context_len, max_question_len, drop_prob=0.1):
        super(DCNAttention, self).__init__()
        self.drop_prob = drop_prob

        self.max_question_len = max_question_len
        self.max_context_len = max_context_len

        self.W_prime_layer = nn.Linear(
            in_features = hidden_size,
            out_features = hidden_size,
            bias = True
        )
        nn.init.xavier_uniform_(self.W_prime_layer.weight,gain=1)

        self.c_sentinel = nn.Parameter(torch.zeros(1,hidden_size))
        nn.init.xavier_uniform_(self.c_sentinel)

        self.q_sentinel = nn.Parameter(torch.zeros(1,hidden_size))
        nn.init.xavier_uniform_(self.q_sentinel)

        self.lstmEncoder= torch.nn.LSTM(
            input_size = 2 * hidden_size,
            hidden_size = 2 * hidden_size,
            num_layers = 1,
            bias = True,
            batch_first = True,
            dropout = self.drop_prob,
            bidirectional = True
        )
        """
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))
        """
    """
        Forward pass of DCN attention layer.

        @param c: tensor of floats, shape (batch_size, c_len, hidden_size)
        @param q: tensor of floats, shape (batch_size, q_len, hidden_size)

        @returns x:  shape (batch_size, c_len, hidden_size * 8)
    """
    def forward(self, c, q, c_mask, q_mask, device):
        batch_size, c_len, hidden_size = c.size()
        q_len = q.size(1)

        N = c_len
        M = q_len
        #print("batch_size: ", batch_size)
        #print("N/c_len : ", N)
        #print("M/q_len : ", M)

        ### PROJECTED HIDDEN STATES ###
        q_inter = torch.cat([q, torch.zeros((batch_size, self.max_question_len - q_len, hidden_size), device=device)], dim=1) # (bs, max_question_len, hidden_size)
        #print("q_inter.size(): ", q_inter.size())
        q_prime_full = torch.tanh(self.W_prime_layer(q_inter)) # (bs, max_question_len, hidden_size)
        #print("q_prime_full.size(): ", q_prime_full.size())
        q_prime = q_prime_full[:,:M,:] # (bs, M, hidden_size)
        #print("q_prime.size(): ", q_prime.size())

        ### ADDING SENTINEL STATES ###
        c_sentinel_batch = torch.ones((batch_size, self.c_sentinel.size(0), self.c_sentinel.size(1)),  device=device) * self.c_sentinel
        q_sentinel_batch = torch.ones((batch_size, self.q_sentinel.size(0), self.q_sentinel.size(1)),  device=device) * self.q_sentinel
        c_new = torch.cat([c, c_sentinel_batch], dim=1) # (bs, N+1, hidden_size)
        q_new = torch.cat([q_prime, q_sentinel_batch], dim=1) # (bs, M+1, hidden_size)

        ### AFFINITY MATRIX: L ###
        row_of_zeros = torch.zeros((batch_size,1), device=device) # (bs, 1)
        adj_c_mask = torch.cat([c_mask.float(), row_of_zeros], dim = 1) # (bs, N + 1)
        adj_q_mask = torch.cat([q_mask.float(), row_of_zeros], dim = 1) # (bs, M + 1)
        adj_c_mask_unsqueezed = torch.unsqueeze(adj_c_mask, dim = -1) # (bs, N + 1, 1)
        adj_q_mask_unsqueezed = torch.unsqueeze(adj_q_mask, dim = 1) # (bs, 1, M + 1)
        L_mask = adj_c_mask_unsqueezed * adj_q_mask_unsqueezed # (bs, N + 1, M+1)
        q_new_perm = q_new.permute(0,2,1) # (bs, hidden_size, M+1)
        L = torch.bmm(c_new,q_new_perm) # (bs, N+1, M+1)

        ### C2Q ###
        alpha = masked_softmax(L, L_mask, dim=2) # (bs, N+1, M+1)
        a = torch.bmm(alpha,q_new) # (bs, N+1, hidden_size)

        ### Q2C ###
        beta = masked_softmax(L, L_mask, dim=1) # (bs, N+1, M+1)
        beta_perm = beta.permute(0,2,1) # (bs, M+1, N+1)
        b = torch.bmm(beta_perm,c_new) # (bs, M+1, hidden_size)

        ### 2ND LEVEL ATTN ###
        s_all = torch.bmm(alpha,b) # (bs, N+1, hidden_size)
        s = s_all[:,:-1,:] # (bs, N, hidden_size)

        ### ENCODED OUTPUT ###
        a_cut = a[:, :N, :] # (bs, N, hidden_size)
        encoder_input = torch.cat([s, a_cut], dim=2) # (bs, N, 2*hidden_size)

        encoder_output,_ = self.lstmEncoder(encoder_input) # (bs, N, 4*hidden_size)
        assert encoder_output.size() == (batch_size, N, 4*hidden_size)

        dropout_layer = torch.nn.Dropout(self.drop_prob)
        encoder_output = dropout_layer(encoder_output)

        return encoder_output

        """
        batch_size, c_len, _ = c.size()

        q_len = q.size(1)

        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)

        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x
        """


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        """
        print("batch_size: ", batch_size)
        print("c_len: ", c_len)
        """
        q_len = q.size(1)
        """
        print("q_len: ", q_len)
        print("c.size(): ", c.size())
        print("q.size(): ", q.size())
        """
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        """
        print("s.size() ", s.size())
        print("c_mask.size() :", c_mask.size())
        print("q_mask.size() :", q_mask.size())
        """
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
