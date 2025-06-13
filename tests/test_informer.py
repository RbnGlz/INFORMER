import importlib.util
import os
import sys
import types
import torch
import torch.nn as nn


def load_informer():
    # Create stub packages and modules
    nf = types.ModuleType("neuralforecast")
    nf.common = types.ModuleType("neuralforecast.common")
    nf.common._modules = types.ModuleType("neuralforecast.common._modules")
    nf.common._base_model = types.ModuleType("neuralforecast.common._base_model")
    nf.losses = types.ModuleType("neuralforecast.losses")
    nf.losses.pytorch = types.ModuleType("neuralforecast.losses.pytorch")

    sys.modules["neuralforecast"] = nf
    sys.modules["neuralforecast.common"] = nf.common
    sys.modules["neuralforecast.common._modules"] = nf.common._modules
    sys.modules["neuralforecast.common._base_model"] = nf.common._base_model
    sys.modules["neuralforecast.losses"] = nf.losses
    sys.modules["neuralforecast.losses.pytorch"] = nf.losses.pytorch

    class DataEmbedding(nn.Module):
        def __init__(self, c_in, exog_input_size, hidden_size, **kwargs):
            super().__init__()
            self.linear = nn.Linear(c_in + exog_input_size, hidden_size)

        def forward(self, x, exog=None):
            if exog is not None and exog.numel() > 0:
                x = torch.cat([x, exog], dim=-1)
            return self.linear(x)

    class TransEncoderLayer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, *args, **kwargs):
            return x

    class TransEncoder(nn.Module):
        def __init__(self, layers, *args, **kwargs):
            super().__init__()
            self.layers = nn.ModuleList(layers)

        def forward(self, x, *args, **kwargs):
            for layer in self.layers:
                x = layer(x)
            return x, None

    class TransDecoderLayer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, enc, *args, **kwargs):
            return x

    class TransDecoder(nn.Module):
        def __init__(self, layers, norm_layer=None, projection=None):
            super().__init__()
            self.layers = nn.ModuleList(layers)
            self.projection = projection

        def forward(self, x, enc, *args, **kwargs):
            for layer in self.layers:
                x = layer(x, enc)
            if self.projection is not None:
                x = self.projection(x)
            return x

    class AttentionLayer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, *args, **kwargs):
            return x

    nf.common._modules.TransEncoderLayer = TransEncoderLayer
    nf.common._modules.TransEncoder = TransEncoder
    nf.common._modules.TransDecoderLayer = TransDecoderLayer
    nf.common._modules.TransDecoder = TransDecoder
    nf.common._modules.DataEmbedding = DataEmbedding
    nf.common._modules.AttentionLayer = AttentionLayer

    class MAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.outputsize_multiplier = 1

    nf.losses.pytorch.MAE = MAE

    class BaseModel(nn.Module):
        def __init__(self, h, input_size, futr_exog_list=None, loss=None, **kwargs):
            super().__init__()
            self.h = h
            self.input_size = input_size
            self.futr_exog_list = futr_exog_list or []
            self.futr_exog_size = len(self.futr_exog_list)
            self.loss = loss or MAE()

    nf.common._base_model.BaseModel = BaseModel

    spec = importlib.util.spec_from_file_location(
        "neuralforecast.models.informer",
        os.path.join(os.path.dirname(__file__), os.pardir, "informer.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Informer


Informer = load_informer()

def test_forward_output_shape():
    batch_size = 2
    input_size = 5
    h = 2
    model = Informer(h=h, input_size=input_size)

    windows_batch = {
        "insample_y": torch.randn(batch_size, input_size, 1),
        "futr_exog": torch.zeros(batch_size, input_size + h, 0),
    }

    out = model.forward(windows_batch)
    assert out.shape == (batch_size, h, 1)
