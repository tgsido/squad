import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttn(nn.Module):
    """ SimpleAttn:
    """
    def __init__(self, hidden_size, drop_prob):
        """
        @param
        """
        super(SimpleAttn, self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob




    def forward(self, values, value_mask, keys):
        """ Forward pass of SimpleAttn module.
        @param values: tensor of floats, shape # (batch_size, M, hidden_size)
        @param value_mask: tensor of floats, shape # (batch_size, M)
        @param keys: tensor of floats, shape # (batch_size, N, hidden_size)

        @returns logits: tensor of floats, shape # (batch_size, N, M)
        @returns attn_dist: tensor of floats, shape # (batch_size, N, M)
        @returns attn_output: tensor of floats, shape # (batch_size, N, hidden_size)
        """
        def softmax_with_mask(logits, mask, dim):
            mask = mask.type(torch.float32)
            masked_logits = mask * logits + (1 - mask) * -1e30
            softmax_fn = F.softmax
            prob_dist = softmax_fn(masked_logits, dim)
            return masked_logits, prob_dist

        batch_size, M, hidden_size = values.size()
        _, N, _ = keys.size()

        values_perm = values.view(batch_size, hidden_size, M)
        assert values_perm.size() == (batch_size, hidden_size, M)

        attn_logits = torch.bmm(keys,values_perm)
        assert attn_logits.size() == (batch_size, N, M)

        attn_logits_mask = value_mask.unsqueeze(1)
        assert attn_logits_mask.size() == (batch_size, 1, M)

        attn_logits, attn_dist = softmax_with_mask(attn_logits, attn_logits_mask, dim = 2)
        assert attn_logits.size() == (batch_size, N, M)
        assert attn_dist.size() == (batch_size, N, M)

        attn_output = torch.bmm(attn_dist,values)
        assert attn_output.size() == (batch_size, N, hidden_size)

        dropout_layer = nn.Dropout(p = self.drop_prob)
        attn_output = dropout_layer(attn_output)

        return attn_logits, attn_dist, attn_output
