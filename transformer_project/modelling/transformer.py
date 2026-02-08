import torch
import torch.nn as nn

from .embedding import Embeddings
from .positional_encoding import PositionalEncoding
from .functional import BaseTransformerLayer, TransformerDecoderLayer

class Transformer(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        n_heads: int, 
        num_encoder_layers: int, 
        num_decoder_layers: int, 
        dim_feedforward: int, 
        dropout: float = 0.1, 
        max_len: int = 5000,
        positional_encoding_type: str = "sinusoidal",  # "sinusoidal" or "rope"
        use_gqa: bool = False,
        num_kv_heads: int = None  # Number of KV heads for GQA (defaults to n_heads if None)
    ):
        super().__init__()
        
        self.positional_encoding_type = positional_encoding_type
        self.use_gqa = use_gqa
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else n_heads
        self.n_heads = n_heads
        self.d_model = d_model
        
        use_rope = (positional_encoding_type == "rope")

        self.embedding = Embeddings(d_model, vocab_size)
        
        if use_rope:
            self.pos_encoding = nn.Dropout(dropout)
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.encoder_layers = nn.ModuleList([
            BaseTransformerLayer(
                input_dim=d_model, 
                num_heads=n_heads, 
                feature_dim=dim_feedforward, 
                dropout=dropout,
                use_rope=use_rope,
                max_len=max_len,
                use_gqa=use_gqa,
                num_kv_heads=self.num_kv_heads
            )
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                input_dim=d_model, 
                num_heads=n_heads, 
                feature_dim=dim_feedforward, 
                dropout=dropout,
                use_rope=use_rope,
                max_len=max_len,
                use_gqa=use_gqa,
                num_kv_heads=self.num_kv_heads
            )
            for _ in range(num_decoder_layers)
        ])
        
        self.projection = nn.Linear(d_model, vocab_size, bias=False)
        
        self.projection.weight = self.embedding.lut.weight
    
    def get_attention_type(self) -> str:
        if self.use_gqa:
            return f"GQA (Q={self.n_heads}, KV={self.num_kv_heads})"
        else:
            return f"MHA (heads={self.n_heads})"
    
    def get_kv_cache_info(self, batch_size: int, seq_len: int) -> dict:
        d_k = self.d_model // self.n_heads
        
        mha_kv_cache_per_layer = 2 * batch_size * self.n_heads * seq_len * d_k
        
        gqa_kv_cache_per_layer = 2 * batch_size * self.num_kv_heads * seq_len * d_k
        
        num_encoder = len(self.encoder_layers)
        num_decoder = len(self.decoder_layers)
        
        actual_cache = gqa_kv_cache_per_layer * (num_encoder + 2 * num_decoder)
        mha_equivalent = mha_kv_cache_per_layer * (num_encoder + 2 * num_decoder)
        
        return {
            "actual_kv_cache_elements": actual_cache,
            "mha_equivalent_elements": mha_equivalent,
            "memory_reduction_ratio": actual_cache / mha_equivalent if mha_equivalent > 0 else 1.0,
            "memory_savings_percent": (1 - actual_cache / mha_equivalent) * 100 if mha_equivalent > 0 else 0.0
        }

    def encode(self, src, src_mask=None):
        x = self.embedding(src)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt)
        x = self.pos_encoding(x)
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, encoder_mask=src_mask, attention_mask=tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        
        output = self.decode(tgt, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        
        logits = self.projection(output)
        return logits
    
    @torch.no_grad()
    def translate(self, src: torch.Tensor, src_mask: torch.Tensor = None,
                  max_len: int = 100, bos_idx: int = 1, eos_idx: int = 2) -> torch.Tensor:
        return self.generate(src, src_mask, max_len, bos_idx, eos_idx)
