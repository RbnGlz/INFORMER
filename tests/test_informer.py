import torch
import pytest

from informer import Informer


def test_forward_shape():
    batch_size = 2
    h = 2
    input_size = 4
    model = Informer(h=h, input_size=input_size)
    insample_y = torch.randn(batch_size, input_size, 1)
    futr_exog = torch.empty(batch_size, input_size + h, 0)
    batch = {"insample_y": insample_y, "futr_exog": futr_exog}
    out = model(batch)
    assert out.shape == (batch_size, h, 1)


def test_invalid_decoder_multiplier():
    with pytest.raises(Exception):
        Informer(h=2, input_size=4, decoder_input_size_multiplier=1.2)


def test_invalid_activation():
    with pytest.raises(Exception):
        Informer(h=2, input_size=4, activation="tanh")
