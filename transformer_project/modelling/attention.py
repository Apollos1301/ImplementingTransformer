import torch
import torch.nn as nn
import math
from typing import Optional
from .positional_encoding import RotaryEmbedding, apply_rotary_pos_emb


class Attention(nn.Module):
    def __init__(self, mask_future=False):
        super().__init__()
        self.mask_future = mask_future

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
             mask_expanded = mask.unsqueeze(1)
             scores = scores.masked_fill(mask_expanded == 0, -1e9)

        if self.mask_future:
            seq_len_q = query.size(1)
            seq_len_k = key.size(1)
            causal_mask = torch.tril(torch.ones((seq_len_q, seq_len_k), device=scores.device)).bool()
            
            scores = scores.masked_fill(causal_mask.unsqueeze(0) == 0, -1e9)
        
        attn_probs = torch.softmax(scores, dim=-1)
        
        output = torch.matmul(attn_probs, value)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, mask_future: bool = False, use_rope: bool = False, max_len: int = 5000):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.mask_future = mask_future
        self.use_rope = use_rope
        
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        
        self.output_transform = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        if self.use_rope:
             self.rotary_emb = RotaryEmbedding(self.d_k, max_seq_len=max_len)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
             if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
             
             attn_scores = attn_scores.masked_fill(mask == 0, -1e4)
        
        if self.mask_future:
            seq_len_q = Q.size(-2)
            seq_len_k = K.size(-2)
            causal_mask = torch.tril(torch.ones((seq_len_q, seq_len_k), device=attn_scores.device)).bool()
            attn_scores = attn_scores.masked_fill(causal_mask == 0, -1e4)
        
        # 4. Softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # 5. Dropout
        attn_probs = self.dropout(attn_probs)
        
        # 6. Matmul with V
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.query_transform(query)
        K = self.key_transform(key)
        V = self.value_transform(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        if self.use_rope:
             msg_seq_len = Q.shape[-2]
             cos, sin = self.rotary_emb(Q, msg_seq_len)
             Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.output_transform(output)
        
        return output


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        num_kv_heads: int = None,
        dropout: float = 0.0, 
        mask_future: bool = False, 
        use_rope: bool = False, 
        max_len: int = 5000
    ):
        super().__init__()
        
        if num_kv_heads is None:
            num_kv_heads = num_heads
            
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        self.mask_future = mask_future
        self.use_rope = use_rope
        
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        
        self.kv_dim = num_kv_heads * self.d_k
        self.key_transform = nn.Linear(d_model, self.kv_dim, bias=False)
        self.value_transform = nn.Linear(d_model, self.kv_dim, bias=False)
        
        self.output_transform = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(self.d_k, max_seq_len=max_len)
    
    def _expand_kv_to_match_heads(self, kv: torch.Tensor) -> torch.Tensor:
        batch_size, num_kv_heads, seq_len, d_k = kv.shape
        
        kv = kv.unsqueeze(2).expand(batch_size, num_kv_heads, self.num_queries_per_kv, seq_len, d_k)
        
        kv = kv.reshape(batch_size, self.num_heads, seq_len, d_k)
        
        return kv
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e4)
        
        if self.mask_future:
            seq_len_q = Q.size(-2)
            seq_len_k = K.size(-2)
            causal_mask = torch.tril(torch.ones((seq_len_q, seq_len_k), device=attn_scores.device)).bool()
            attn_scores = attn_scores.masked_fill(causal_mask == 0, -1e4)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        attn_probs = self.dropout(attn_probs)
        
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        Q = self.query_transform(query)
        K = self.key_transform(key)
        V = self.value_transform(value)
        
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        
        K = K.view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        if self.use_rope:
            cos, sin = self.rotary_emb(Q, seq_len_q)
            Q = self._apply_rope_to_tensor(Q, cos, sin)
            K = self._apply_rope_to_tensor(K, cos[:, :, :seq_len_k, :], sin[:, :, :seq_len_k, :])
        
        K = self._expand_kv_to_match_heads(K)
        V = self._expand_kv_to_match_heads(V)
        
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        output = self.output_transform(output)
        
        return output
    
    def _apply_rope_to_tensor(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)
    
    def get_kv_cache_size(self, batch_size: int, seq_len: int) -> int:
        return 2 * batch_size * self.num_kv_heads * seq_len * self.d_k
    
    @staticmethod
    def compare_kv_cache_reduction(num_heads: int, num_kv_heads: int) -> float:
        return num_kv_heads / num_heads
