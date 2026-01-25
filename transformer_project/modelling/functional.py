import torch
import torch.nn as nn
from .attention import MultiHeadAttention


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class BaseTransformerLayer(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, feature_dim: int, 
                 dropout: float = 0.1, mask_future: bool = False):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(
            d_model=input_dim,
            num_heads=num_heads,
            mask_future=mask_future,
            dropout=dropout
        )
        
        self.feature_transformation = PositionWiseFeedForward(
            d_model=input_dim,
            d_ff=feature_dim,
            dropout=dropout
        )
        
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_output = self.self_attention(x, x, x, mask)
        x = self.layer_norm_1(x + self.dropout1(attn_output))
        
        ff_output = self.feature_transformation(x)
        x = self.layer_norm_2(x + self.dropout2(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, feature_dim: int, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(
            d_model=input_dim,
            num_heads=num_heads,
            mask_future=True,
            dropout=dropout
        )
        
        self.encoder_attention = MultiHeadAttention(
            d_model=input_dim,
            num_heads=num_heads,
            mask_future=False,
            dropout=dropout
        )
        
        self.feature_transformation = PositionWiseFeedForward(
            d_model=input_dim,
            d_ff=feature_dim,
            dropout=dropout
        )
        
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.layer_norm_1(x + self.dropout1(attn_output))
        
        cross_attn_output = self.encoder_attention(x, encoder_output, encoder_output, src_mask)
        x = self.layer_norm_2(x + self.dropout2(cross_attn_output))
        
        ff_output = self.feature_transformation(x)
        x = self.layer_norm_3(x + self.dropout3(ff_output))
        
        return x
