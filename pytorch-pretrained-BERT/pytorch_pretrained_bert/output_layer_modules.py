#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import pdb
from attn_modules import SimpleAttn
class AnswerPointerOutput(nn.Module):
    """ AnswerPointerOutput:
    """
    def __init__(self, hidden_size, drop_prob):
        """ Init Answer.
        @param hidden_size (int): dimensionality of final output embeddings
        """
        super(AnswerPointerOutput, self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob

        ### Answer-Pointer RNN ###
        self.rnn = nn.RNN(
            input_size = self.hidden_size,
            hidden_size = self.hidden_size,
            num_layers = 1,
            nonlinearity = 'tanh',
            bias = True,
            batch_first = True,
            bidirectional = False
        )

    def forward(self, sequence_output, attention_mask):
        """ Forward pass of AnswerPointerOutput module.
        @param sequence_output: tensor of floats, shape # (batch_size, sequence_length, hidden_size)
        @returns logits: tensor of floats, shape # (batch_size, sequence_length, 2)
        """
        #print("in ansr-ptr:")
        batch_size, seq_len, hidden_size = sequence_output.size()
        simple_attn = SimpleAttn(hidden_size = hidden_size, drop_prob = self.drop_prob)

        ### 1st RNN Pass ###
        output, h_n = self.rnn(sequence_output)

        assert output.size() == (batch_size, seq_len, hidden_size)
        h_n = h_n.view(batch_size, 1, hidden_size)
        #pdb.set_trace()
        assert h_n.size() == (batch_size, 1, hidden_size)

        attn_logits, attn_dist, attn_output = simple_attn(sequence_output, attention_mask, h_n)
        assert attn_logits.size() == (batch_size, 1, seq_len)
        assert attn_dist.size() == (batch_size, 1, seq_len)
        assert attn_output.size() == (batch_size, 1, hidden_size)

        beta_s = attn_dist.squeeze(1)
        assert beta_s.size() == (batch_size,seq_len)
        a_s = attn_output.squeeze(1).unsqueeze(0)
        assert a_s.size() == (1, batch_size, hidden_size)
        start_logits = attn_logits.squeeze(1).unsqueeze(-1)
        assert start_logits.size() == (batch_size, seq_len, 1)

        ### 2nd RNN Pass ###
        output, h_n = self.rnn(sequence_output, a_s)

        assert output.size() == (batch_size, seq_len, hidden_size)
        h_n = h_n.view(batch_size, 1, hidden_size)
        assert h_n.size() == (batch_size, 1, hidden_size)

        attn_logits, attn_dist, attn_output = simple_attn(sequence_output, attention_mask, h_n)
        assert attn_logits.size() == (batch_size, 1, seq_len)
        assert attn_dist.size() == (batch_size, 1, seq_len)
        assert attn_output.size() == (batch_size, 1, hidden_size)

        beta_e = attn_dist.squeeze(1)
        assert beta_e.size() == (batch_size,seq_len)
        a_e = attn_output.squeeze(1)
        assert a_e.size() == (batch_size, hidden_size)
        end_logits = attn_logits.squeeze(1).unsqueeze(-1)
        assert end_logits.size() == (batch_size, seq_len, 1)

        logits = torch.cat([start_logits, end_logits], dim=-1) # (batch_size, sequence_length,2)
        assert logits.size() == (batch_size, seq_len, 2)

        return logits

class AnswerPointerGruOutput(nn.Module):
    """ AnswerPointerGruOutput:
    """
    def __init__(self, hidden_size, drop_prob):
        """ Init Answer.
        @param hidden_size (int): dimensionality of final output embeddings
        """
        super(AnswerPointerOutput, self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob

        ### Answer-Pointer GRU ###
        self.gru = nn.GRU(
            input_size = self.hidden_size,
            hidden_size = self.hidden_size,
            num_layers = 1,
            nonlinearity = 'tanh',
            bias = True,
            batch_first = True,
            bidirectional = False
        )

    def forward(self, sequence_output, attention_mask):
        """ Forward pass of AnswerPointerGruOutput module.
        @param sequence_output: tensor of floats, shape # (batch_size, sequence_length, hidden_size)
        @returns logits: tensor of floats, shape # (batch_size, sequence_length, 2)
        """
        print("in ansr-ptr-gru:")
        batch_size, seq_len, hidden_size = sequence_output.size()
        simple_attn = SimpleAttn(hidden_size = hidden_size, drop_prob = self.drop_prob)

        ### 1st GRU Pass ###
        output, h_n = self.gru(sequence_output)

        assert output.size() == (batch_size, seq_len, hidden_size)
        h_n = h_n.view(batch_size, 1, hidden_size)
        #pdb.set_trace()
        assert h_n.size() == (batch_size, 1, hidden_size)

        attn_logits, attn_dist, attn_output = simple_attn(sequence_output, attention_mask, h_n)
        assert attn_logits.size() == (batch_size, 1, seq_len)
        assert attn_dist.size() == (batch_size, 1, seq_len)
        assert attn_output.size() == (batch_size, 1, hidden_size)

        beta_s = attn_dist.squeeze(1)
        assert beta_s.size() == (batch_size,seq_len)
        a_s = attn_output.squeeze(1).unsqueeze(0)
        assert a_s.size() == (1, batch_size, hidden_size)
        start_logits = attn_logits.squeeze(1).unsqueeze(-1)
        assert start_logits.size() == (batch_size, seq_len, 1)

        ### 2nd GRU Pass ###
        output, h_n = self.gru(sequence_output, a_s)

        assert output.size() == (batch_size, seq_len, hidden_size)
        h_n = h_n.view(batch_size, 1, hidden_size)
        assert h_n.size() == (batch_size, 1, hidden_size)

        attn_logits, attn_dist, attn_output = simple_attn(sequence_output, attention_mask, h_n)
        assert attn_logits.size() == (batch_size, 1, seq_len)
        assert attn_dist.size() == (batch_size, 1, seq_len)
        assert attn_output.size() == (batch_size, 1, hidden_size)

        beta_e = attn_dist.squeeze(1)
        assert beta_e.size() == (batch_size,seq_len)
        a_e = attn_output.squeeze(1)
        assert a_e.size() == (batch_size, hidden_size)
        end_logits = attn_logits.squeeze(1).unsqueeze(-1)
        assert end_logits.size() == (batch_size, seq_len, 1)

        logits = torch.cat([start_logits, end_logits], dim=-1) # (batch_size, sequence_length,2)
        assert logits.size() == (batch_size, seq_len, 2)

        return logits

class RNNCNNOutput(nn.Module):
    """ AnswerPointerOutput:
        Module that uses two RNNs to convert
    """
    def __init__(self, hidden_size):
        """ Init Answer.
        @param hidden_size (int): dimensionality of final output embeddings
        """
        super(RNNCNNOutput, self).__init__()
        self.hidden_size = hidden_size

        ###  RNN ###
        self.rnn = nn.RNN(
            input_size = self.hidden_size,
            hidden_size = self.hidden_size,
            num_layers = 1,
            nonlinearity = 'tanh',
            bias = True,
            batch_first = True,
            bidirectional = False
        )

        self.proj_down = nn.Linear(
            in_features = self.hidden_size * 2,
            out_features = 2,
            bias = True
        )
        nn.init.xavier_uniform_(self.proj_down.weight)

        self.conv_layer = nn.Conv1d(
            in_channels = 384,
            out_channels = 384,
            kernel_size = hidden_size - 1,
            padding = 0,
            stride = 1,
            bias = True
        )

    def forward(self, sequence_output):
        """ Forward pass of RNNCNNOutput module.
        @param sequence_output: tensor of floats, shape # (batch_size, sequence_length, hidden_size)
        @returns logits: tensor of floats, shape # (batch_size, sequence_length, 2)
        """
        batch_size, seq_len, hidden_size = sequence_output.size()
        beta_s, a = self.rnn(sequence_output)
        assert beta_s.size() == (batch_size, seq_len, hidden_size)
        assert a.size() == (batch_size, 1, hidden_size)
        beta_e, _ = self.rnn(sequence_output, a)
        assert beta_e.size() == (batch_size, seq_len, hidden_size)

        logits_rnn = torch.cat([beta_s, beta_e], dim=-1) # (batch_size, sequence_length, 2*hidden_size)
        assert logits_rnn.size() == (batch_size, seq_len, 2*hidden_size)
        logits_rnn = self.proj_down(logits_rnn) # (batch_size, sequence_length, 2)
        assert logits_rnn.size() == (batch_size, seq_len, 2)

        logits_conv = self.conv_layer(sequence_output) # (batch_size, sequence_length, 2)
        assert logits_conv.size() == (batch_size, seq_len, 2)

        logits = logits_conv + logits_rnn # (batch_size, sequence_length, 2)
        assert logits.size() == (batch_size, seq_len, 2)
        return logits
        """
        x_proj = torch.nn.functional.relu(self.proj(x_conv_out)) # (sentence_length * batch_size, e_word)
        x_gate = torch.sigmoid(self.gate(x_conv_out)) # (sentence_length * batch_size, e_word)
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out # (sentence_length * batch_size, e_word)
        """
