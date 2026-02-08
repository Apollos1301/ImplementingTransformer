import torch.nn as nn
from .attention import MultiHeadAttention, GroupedQueryAttention
from .feedforward import PositionWiseFeedForward
from .layer_norm import LayerNorm


def get_attention_module(
    d_model: int, 
    num_heads: int, 
    dropout: float, 
    mask_future: bool = False,
    use_rope: bool = False, 
    max_len: int = 5000,
    use_gqa: bool = False,
    num_kv_heads: int = None
):
    if use_gqa:
        return GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            mask_future=mask_future,
            use_rope=use_rope,
            max_len=max_len
        )
    else:
        return MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            mask_future=mask_future,
            use_rope=use_rope,
            max_len=max_len
        )


class BaseTransformerLayer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        num_heads: int, 
        feature_dim: int, 
        dropout: float = 0.1, 
        use_rope: bool = False, 
        max_len: int = 5000,
        use_gqa: bool = False,
        num_kv_heads: int = None
    ):
        super().__init__()
        
        d_model = input_dim
        d_ff = feature_dim
        
        self.self_attention = get_attention_module(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            mask_future=False,
            use_rope=use_rope,
            max_len=max_len,
            use_gqa=use_gqa,
            num_kv_heads=num_kv_heads
        )
        self.feature_transformation = PositionWiseFeedForward(d_model, d_ff)
        
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        
        attn_output = self.self_attention(x, x, x, mask)
        x = self.layer_norm_1(x + self.dropout1(attn_output))
        
        ff_output = self.feature_transformation(x)
        x = self.layer_norm_2(x + self.dropout2(ff_output))
        
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        num_heads: int, 
        feature_dim: int, 
        dropout: float = 0.1, 
        use_rope: bool = False, 
        max_len: int = 5000,
        use_gqa: bool = False,
        num_kv_heads: int = None
    ):
        super().__init__()
        
        d_model = input_dim
        d_ff = feature_dim
        
        self.self_attention = get_attention_module(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            mask_future=True,
            use_rope=use_rope,
            max_len=max_len,
            use_gqa=use_gqa,
            num_kv_heads=num_kv_heads
        )
        
        self.encoder_attention = get_attention_module(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            mask_future=False,
            use_rope=False,
            max_len=max_len,
            use_gqa=use_gqa,
            num_kv_heads=num_kv_heads
        )
        
        self.feature_transformation = PositionWiseFeedForward(d_model, d_ff)
        
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.layer_norm_3 = LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, encoder_mask=None, attention_mask=None):
        
        attn1 = self.self_attention(x, x, x, attention_mask)
        x = self.layer_norm_1(x + self.dropout1(attn1))
        
        attn2 = self.encoder_attention(x, encoder_output, encoder_output, encoder_mask)
        x = self.layer_norm_2(x + self.dropout2(attn2))
        
        ff = self.feature_transformation(x)
        x = self.layer_norm_3(x + self.dropout3(ff))
        
        return x
