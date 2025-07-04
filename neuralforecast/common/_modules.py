import torch
from torch import nn

class DataEmbedding(nn.Module):
    """Embed input series and optional exogenous features.

    Parameters
    ----------
    c_in : int
        Number of features of the main time series.
    exog_input_size : int, optional
        Size of the exogenous inputs concatenated to ``x``.
    hidden_size : int, optional
        Dimension of the resulting embedding.
    pos_embedding : bool, optional
        Unused flag kept for API compatibility.
    dropout : float, optional
        Dropout rate, currently unused.
    """

    def __init__(self, c_in, exog_input_size=0, hidden_size=16, pos_embedding=True, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(c_in + exog_input_size, hidden_size)
    def forward(self, x, x_mark=None):
        if x_mark is not None:
            x = torch.cat([x, x_mark], dim=-1)
        return self.proj(x)

class AttentionLayer(nn.Module):
    """Thin wrapper around an attention module.

    Parameters
    ----------
    attention : nn.Module
        Module implementing the attention operation.
    d_model : int
        Embedding dimension of the model (kept for compatibility).
    n_heads : int
        Number of attention heads (kept for compatibility).
    """

    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        self.attention = attention

    def forward(self, queries, keys, values, attn_mask=None):
        return self.attention(queries, keys, values, attn_mask)

class TransEncoderLayer(nn.Module):
    """Single encoder block based on self-attention.

    Parameters
    ----------
    attn_layer : nn.Module
        Attention layer applied to the input sequence.
    d_model : int
        Embedding dimension of the model.
    conv_hidden_size : int
        Size of intermediate convolutional layers (unused here).
    dropout : float, optional
        Dropout rate used inside the layer (unused).
    activation : str, optional
        Name of the activation function (unused).
    """

    def __init__(self, attn_layer, d_model, conv_hidden_size, dropout=0.1, activation="gelu"):
        super().__init__()
        self.attn_layer = attn_layer

    def forward(self, x, attn_mask=None):
        x, _ = self.attn_layer(x, x, x, attn_mask)
        return x, None

class TransEncoder(nn.Module):
    """Stack of encoder layers used by Informer.

    Parameters
    ----------
    layers : list[nn.Module]
        Sequence of :class:`TransEncoderLayer` objects.
    conv_layers : list[nn.Module], optional
        Optional convolutional layers for distillation.
    norm_layer : nn.Module, optional
        Normalization layer applied after the stack.
    """

    def __init__(self, layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x, _ = layer(x, attn_mask)
        return x, None

class TransDecoderLayer(nn.Module):
    """Decoder block with self- and cross-attention.

    Parameters
    ----------
    self_attn_layer : nn.Module
        Attention layer applied to the decoder inputs.
    cross_attn_layer : nn.Module
        Attention layer attending to the encoder outputs (unused).
    d_model : int
        Embedding dimension of the model.
    conv_hidden_size : int
        Size of intermediate convolutional layers (unused).
    dropout : float, optional
        Dropout rate for the layer (unused).
    activation : str, optional
        Name of the activation function (unused).
    """

    def __init__(self, self_attn_layer, cross_attn_layer, d_model, conv_hidden_size, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn_layer = self_attn_layer

    def forward(self, x, enc_out, x_mask=None, cross_mask=None):
        x, _ = self.self_attn_layer(x, x, x, x_mask)
        return x

class TransDecoder(nn.Module):
    """Stack of decoder layers producing the final predictions.

    Parameters
    ----------
    layers : list[nn.Module]
        Sequence of :class:`TransDecoderLayer` modules.
    norm_layer : nn.Module, optional
        Optional normalization applied after the stack.
    projection : nn.Module, optional
        Linear projection applied to the decoder output.
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.projection = projection if projection is not None else nn.Identity()
    def forward(self, x, enc_out, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, x_mask, cross_mask)
        x = self.projection(x)
        return x
