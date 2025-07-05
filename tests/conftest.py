"""Configuración compartida para tests."""

import pytest
import torch
import numpy as np
from typing import Dict, Any


@pytest.fixture
def sample_batch() -> Dict[str, torch.Tensor]:
    """Genera un batch de muestra para testing."""
    batch_size, input_size, h = 4, 96, 24
    return {
        "insample_y": torch.randn(batch_size, input_size, 1),
        "futr_exog": torch.randn(batch_size, input_size + h, 3),
    }


@pytest.fixture
def sample_batch_no_exog() -> Dict[str, torch.Tensor]:
    """Genera un batch de muestra sin variables exógenas."""
    batch_size, input_size, h = 4, 96, 24
    return {
        "insample_y": torch.randn(batch_size, input_size, 1),
        "futr_exog": torch.empty(batch_size, input_size + h, 0),
    }


@pytest.fixture
def model_config() -> Dict[str, Any]:
    """Configuración estándar del modelo."""
    return {
        "h": 24,
        "input_size": 96,
        "hidden_size": 64,
        "encoder_layers": 2,
        "decoder_layers": 1,
        "n_head": 4,
        "factor": 3,
        "dropout": 0.1,
        "max_steps": 10,  # Reducido para tests rápidos
    }


@pytest.fixture
def small_model_config() -> Dict[str, Any]:
    """Configuración pequeña del modelo para tests rápidos."""
    return {
        "h": 4,
        "input_size": 16,
        "hidden_size": 32,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "n_head": 2,
        "factor": 2,
        "dropout": 0.0,
        "max_steps": 5,
    }


@pytest.fixture(autouse=True)
def set_random_seed():
    """Fija semilla para reproducibilidad."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)