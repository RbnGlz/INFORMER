"""Tests de integración end-to-end."""

import pytest
import torch
from src.informer.models.informer import Informer


class TestInformerIntegration:
    """Tests de integración completos."""
    
    def test_model_instantiation(self, model_config):
        """Verifica que el modelo se pueda instanciar correctamente."""
        model = Informer(**model_config)
        assert model is not None
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        
    def test_forward_pass(self, model_config, sample_batch):
        """Verifica que el forward pass funcione correctamente."""
        model = Informer(**model_config)
        output = model(sample_batch)
        
        expected_shape = (sample_batch["insample_y"].shape[0], model_config["h"], 1)
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
        
    def test_forward_pass_no_exog(self, model_config, sample_batch_no_exog):
        """Verifica funcionamiento sin variables exógenas."""
        model = Informer(**model_config)
        output = model(sample_batch_no_exog)
        
        expected_shape = (sample_batch_no_exog["insample_y"].shape[0], model_config["h"], 1)
        assert output.shape == expected_shape
        
    def test_training_loop(self, small_model_config, sample_batch):
        """Verifica que el modelo pueda entrenarse."""
        model = Informer(**small_model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        initial_loss = None
        final_loss = None
        
        # Simular loop de entrenamiento
        for i in range(5):
            optimizer.zero_grad()
            output = model(sample_batch)
            loss = output.mean()  # Loss simplificado para testing
            
            if i == 0:
                initial_loss = loss.item()
            final_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            
        assert initial_loss is not None
        assert final_loss is not None
        assert final_loss < float('inf')
        
    def test_inference_reproducibility(self, model_config, sample_batch):
        """Verifica reproducibilidad en inferencia."""
        torch.manual_seed(42)
        model1 = Informer(**model_config)
        
        torch.manual_seed(42)
        output1 = model1(sample_batch)
        
        torch.manual_seed(42)
        model2 = Informer(**model_config)
        
        torch.manual_seed(42)
        output2 = model2(sample_batch)
        
        torch.testing.assert_close(output1, output2, rtol=1e-4, atol=1e-6)
        
    def test_gradient_flow(self, small_model_config, sample_batch):
        """Verifica que los gradientes fluyan correctamente."""
        model = Informer(**small_model_config)
        
        # Verificar que no hay gradientes inicialmente
        for param in model.parameters():
            assert param.grad is None
            
        output = model(sample_batch)
        loss = output.mean()
        loss.backward()
        
        # Verificar que los gradientes existen y no son cero
        gradient_exists = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                gradient_exists = True
                break
                
        assert gradient_exists, "No se encontraron gradientes no ceros"
        
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_different_batch_sizes(self, small_model_config, batch_size):
        """Verifica funcionamiento con diferentes tamaños de batch."""
        model = Informer(**small_model_config)
        
        input_size = small_model_config["input_size"]
        h = small_model_config["h"]
        
        batch = {
            "insample_y": torch.randn(batch_size, input_size, 1),
            "futr_exog": torch.empty(batch_size, input_size + h, 0),
        }
        
        output = model(batch)
        assert output.shape == (batch_size, h, 1)
        
    def test_invalid_decoder_multiplier(self):
        """Verifica que se lance error con multiplicador inválido."""
        with pytest.raises(ValueError, match="decoder_input_size_multiplier"):
            Informer(h=2, input_size=4, decoder_input_size_multiplier=1.2)
            
    def test_invalid_activation(self):
        """Verifica que se lance error con activación inválida."""
        with pytest.raises(ValueError, match="activation"):
            Informer(h=2, input_size=4, activation="tanh")
            
    def test_model_eval_mode(self, model_config, sample_batch):
        """Verifica que el modelo funcione en modo evaluación."""
        model = Informer(**model_config)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_batch)
            
        expected_shape = (sample_batch["insample_y"].shape[0], model_config["h"], 1)
        assert output.shape == expected_shape
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA no disponible")
    def test_cuda_compatibility(self, small_model_config):
        """Verifica compatibilidad con CUDA."""
        model = Informer(**small_model_config).cuda()
        
        input_size = small_model_config["input_size"]
        h = small_model_config["h"]
        
        batch = {
            "insample_y": torch.randn(2, input_size, 1, device="cuda"),
            "futr_exog": torch.empty(2, input_size + h, 0, device="cuda"),
        }
        
        output = model(batch)
        assert output.device.type == "cuda"
        assert output.shape == (2, h, 1)