import torch.nn as nn
import torch.nn.functional as F
import torch
from rotary_embedding_torch import RotaryEmbedding
import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.key_embs = RotaryEmbedding(hidden_dim)
        self.query_embs = RotaryEmbedding(hidden_dim)

    def forward(self, query, key, value, mask=None, dropout=None):
        query = self.query_embs.rotate_queries_or_keys(query)
        key = self.key_embs.rotate_queries_or_keys(key)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -6e4)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
