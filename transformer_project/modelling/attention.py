import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def __init__(self, mask_future: bool = False, dropout: float = 0.0):
        super().__init__()
        self.mask_future = mask_future
        self.dropout = nn.Dropout(dropout)

    def _create_causal_mask(self, seq_len_q: int, seq_len_k: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len_q, seq_len_k, device=device))
        return mask

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            padding_mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(padding_mask == 0, float('-inf'))

        if self.mask_future:
            seq_len_q, seq_len_k = query.size(-2), key.size(-2)
            causal_mask = self._create_causal_mask(seq_len_q, seq_len_k, query.device)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, value)

        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, mask_future: bool = False, dropout: float = 0.0):
        super().__init__()
        self.mask_future = mask_future
        self.dropout = nn.Dropout(dropout)

    def _create_causal_mask(self, seq_len_q: int, seq_len_k: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len_q, seq_len_k, device=device))

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        if self.mask_future:
            seq_len_q, seq_len_k = query.size(-2), key.size(-2)
            causal_mask = self._create_causal_mask(seq_len_q, seq_len_k, query.device)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, value)
        
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mask_future: bool = False, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.mask_future = mask_future
        
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(mask_future=mask_future, dropout=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        Q = self.query_transform(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key_transform(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value_transform(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
        
        attn_output = self.attention(Q, K, V, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.output_transform(attn_output)
        
        return output
