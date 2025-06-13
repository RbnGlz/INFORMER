import torch
from torch import nn

class DataEmbedding(nn.Module):
    def __init__(self, c_in, exog_input_size=0, hidden_size=16, pos_embedding=True, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(c_in + exog_input_size, hidden_size)
    def forward(self, x, x_mark=None):
        if x_mark is not None:
            x = torch.cat([x, x_mark], dim=-1)
        return self.proj(x)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        self.attention = attention

    def forward(self, queries, keys, values, attn_mask=None):
        return self.attention(queries, keys, values, attn_mask)

class TransEncoderLayer(nn.Module):
    def __init__(self, attn_layer, d_model, conv_hidden_size, dropout=0.1, activation="gelu"):
        super().__init__()
        self.attn_layer = attn_layer

    def forward(self, x, attn_mask=None):
        x, _ = self.attn_layer(x, x, x, attn_mask)
        return x, None

class TransEncoder(nn.Module):
    def __init__(self, layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x, _ = layer(x, attn_mask)
        return x, None

class TransDecoderLayer(nn.Module):
    def __init__(self, self_attn_layer, cross_attn_layer, d_model, conv_hidden_size, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn_layer = self_attn_layer

    def forward(self, x, enc_out, x_mask=None, cross_mask=None):
        x, _ = self.self_attn_layer(x, x, x, x_mask)
        return x

class TransDecoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.projection = projection if projection is not None else nn.Identity()
    def forward(self, x, enc_out, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, x_mask, cross_mask)
        x = self.projection(x)
        return x
