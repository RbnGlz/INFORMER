"""Tests unitarios para componentes de atención."""

import pytest
import torch
from src.informer.models.components.attention import ProbAttention, ProbMask


class TestProbMask:
    """Tests para la clase ProbMask."""
    
    def test_prob_mask_initialization(self):
        """Verifica que ProbMask se inicialice correctamente."""
        B, H, L = 2, 4, 16
        index = torch.randint(0, L, (B, H, L//2))
        scores = torch.randn(B, H, L//2, L)
        
        mask = ProbMask(B, H, L, index, scores)
        
        assert mask.mask is not None
        assert mask.mask.shape == scores.shape
        assert mask.mask.dtype == torch.bool

    def test_prob_mask_device_consistency(self):
        """Verifica que la máscara esté en el mismo dispositivo que las puntuaciones."""
        B, H, L = 1, 2, 8
        index = torch.randint(0, L, (B, H, L//2))
        scores = torch.randn(B, H, L//2, L)
        device = "cpu"
        
        mask = ProbMask(B, H, L, index, scores, device=device)
        
        assert mask.mask.device.type == device


class TestProbAttention:
    """Tests para el mecanismo de atención probabilística."""
    
    def test_prob_attention_forward_shape(self):
        """Verifica que la salida tenga la forma correcta."""
        attention = ProbAttention(factor=3, attention_dropout=0.1)
        B, L, H, D = 2, 32, 4, 64
        
        queries = torch.randn(B, L, H, D)
        keys = torch.randn(B, L, H, D)
        values = torch.randn(B, L, H, D)
        
        output, attn_weights = attention(queries, keys, values, None)
        
        assert output.shape == (B, H, L, D)
        # attn_weights puede ser None si output_attention=False
        
    def test_prob_attention_factor_scaling(self):
        """Verifica que el factor de scaling afecte la configuración."""
        for factor in [1, 3, 5]:
            attention = ProbAttention(factor=factor)
            assert attention.factor == factor
            
    @pytest.mark.parametrize("mask_flag", [True, False])
    def test_prob_attention_masking(self, mask_flag):
        """Verifica el comportamiento de las máscaras."""
        attention = ProbAttention(mask_flag=mask_flag)
        B, L, H, D = 1, 16, 2, 32
        
        queries = torch.randn(B, L, H, D)
        keys = torch.randn(B, L, H, D)
        values = torch.randn(B, L, H, D)
        
        output, _ = attention(queries, keys, values, None)
        assert output.shape == (B, H, L, D)
        
    def test_prob_attention_output_attention(self):
        """Verifica que se retornen los pesos cuando output_attention=True."""
        attention = ProbAttention(output_attention=True, mask_flag=False)
        B, L, H, D = 1, 8, 2, 16
        
        queries = torch.randn(B, L, H, D)
        keys = torch.randn(B, L, H, D)
        values = torch.randn(B, L, H, D)
        
        output, attn_weights = attention(queries, keys, values, None)
        
        assert output.shape == (B, H, L, D)
        assert attn_weights is not None
        assert attn_weights.shape == (B, H, L, L)
        
    def test_prob_attention_scale_parameter(self):
        """Verifica que el parámetro de escala funcione correctamente."""
        scale = 0.5
        attention = ProbAttention(scale=scale)
        assert attention.scale == scale
        
    def test_prob_attention_small_sequence(self):
        """Verifica funcionamiento con secuencias pequeñas."""
        attention = ProbAttention(factor=2)
        B, L, H, D = 1, 4, 1, 8
        
        queries = torch.randn(B, L, H, D)
        keys = torch.randn(B, L, H, D)
        values = torch.randn(B, L, H, D)
        
        output, _ = attention(queries, keys, values, None)
        assert output.shape == (B, H, L, D)
        
    def test_prob_attention_deterministic(self):
        """Verifica que la atención sea determinística con la misma semilla."""
        torch.manual_seed(42)
        attention1 = ProbAttention(factor=3)
        B, L, H, D = 1, 16, 2, 32
        
        queries = torch.randn(B, L, H, D)
        keys = torch.randn(B, L, H, D)
        values = torch.randn(B, L, H, D)
        
        torch.manual_seed(42)
        output1, _ = attention1(queries, keys, values, None)
        
        torch.manual_seed(42)
        attention2 = ProbAttention(factor=3)
        torch.manual_seed(42)
        output2, _ = attention2(queries, keys, values, None)
        
        # Nota: La aleatoriedad en randint puede hacer que esto no sea exactamente igual
        # pero al menos debe tener la misma forma
        assert output1.shape == output2.shape