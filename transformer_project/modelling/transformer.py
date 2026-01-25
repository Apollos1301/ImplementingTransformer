import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention
from .functional import BaseTransformerLayer, TransformerDecoderLayer, PositionWiseFeedForward
from .positional_encoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, 
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BaseTransformerLayer(
                input_dim=d_model,
                num_heads=num_heads,
                feature_dim=dim_feedforward,
                dropout=dropout,
                mask_future=False
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                input_dim=d_model,
                num_heads=num_heads,
                feature_dim=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 max_len: int = 5000, pad_idx: int = 0, share_embeddings: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        if share_embeddings:
            self.output_projection.weight = self.embedding.weight
        
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        return self.encoder(src_emb, src_mask)
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        return self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.output_projection(decoder_output)
        return logits
    
    def generate(self, src: torch.Tensor, src_mask: torch.Tensor = None,
                 max_len: int = 100, bos_idx: int = 1, eos_idx: int = 2) -> torch.Tensor:
        batch_size = src.size(0)
        encoder_output = self.encode(src, src_mask)
        
        generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=src.device)
        
        for _ in range(max_len - 1):
            tgt_mask = (generated != self.pad_idx).long()
            decoder_output = self.decode(generated, encoder_output, src_mask, tgt_mask)
            logits = self.output_projection(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == eos_idx).all():
                break
        
        return generated
