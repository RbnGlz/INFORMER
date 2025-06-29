import torch
import pytest

from informer import Informer

# =======================
# Prueba de la salida del modelo
# =======================
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

# ========================================
# Prueba de validación de parámetros inválidos
# ========================================
def test_invalid_decoder_multiplier():
    with pytest.raises(Exception):
        Informer(h=2, input_size=4, decoder_input_size_multiplier=1.2)

def test_invalid_activation():
    with pytest.raises(Exception):
        Informer(h=2, input_size=4, activation="tanh")

# ===========================
# Pruebas adicionales sugeridas
# ===========================

def test_backward_pass():
    model = Informer(h=2, input_size=4)
    insample_y = torch.randn(2, 4, 1, requires_grad=True)
    futr_exog = torch.empty(2, 6, 0)
    batch = {"insample_y": insample_y, "futr_exog": futr_exog}
    output = model(batch)
    loss = output.mean()
    loss.backward()
    assert insample_y.grad is not None

def test_with_exogenous_inputs():
    model = Informer(h=2, input_size=4)
    insample_y = torch.randn(2, 4, 1)
    futr_exog = torch.randn(2, 6, 3)  # 3 variables exógenas
    batch = {"insample_y": insample_y, "futr_exog": futr_exog}
    output = model(batch)
    assert output.shape == (2, 2, 1)

@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_various_batch_sizes(batch_size):
    model = Informer(h=2, input_size=4)
    insample_y = torch.randn(batch_size, 4, 1)
    futr_exog = torch.empty(batch_size, 6, 0)
    batch = {"insample_y": insample_y, "futr_exog": futr_exog}
    output = model(batch)
    assert output.shape == (batch_size, 2, 1)

def test_missing_keys_in_batch():
    model = Informer(h=2, input_size=4)
    insample_y = torch.randn(2, 4, 1)
    with pytest.raises(KeyError):
        batch = {"insample_y": insample_y}  # Falta 'futr_exog'
        model(batch)

def test_cuda_support():
    if torch.cuda.is_available():
        model = Informer(h=2, input_size=4).cuda()
        insample_y = torch.randn(2, 4, 1, device="cuda")
        futr_exog = torch.empty(2, 6, 0, device="cuda")
        batch = {"insample_y": insample_y, "futr_exog": futr_exog}
        output = model(batch)
        assert output.is_cuda

def test_nan_input():
    model = Informer(h=2, input_size=4)
    insample_y = torch.randn(2, 4, 1)
    insample_y[0, 0, 0] = float('nan')
    futr_exog = torch.empty(2, 6, 0)
    batch = {"insample_y": insample_y, "futr_exog": futr_exog}
    with pytest.raises(Exception):
        model(batch)