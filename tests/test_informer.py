"""
Archivo de pruebas unitarias para el modelo Informer.

Este script utiliza pytest y PyTorch para validar el correcto funcionamiento del modelo Informer,
asegurando que cumple con los requisitos de forma, manejo de errores y compatibilidad con diferentes
configuraciones de entrada. Las pruebas incluidas cubren:

- Verificación de la forma de la salida del modelo para distintos tamaños de batch y horizonte de predicción.
- Validación de parámetros inválidos, asegurando que el modelo lanza excepciones apropiadas ante configuraciones erróneas.
- Comprobación de la compatibilidad con el cálculo de gradientes (backpropagation) para entrenamiento supervisado.
- Pruebas con y sin variables exógenas, garantizando que el modelo puede integrar información adicional.
- Manejo de entradas incompletas o con valores no numéricos (NaN), previniendo fallos silenciosos.
- Compatibilidad con ejecución en GPU (CUDA), fundamental para entrenamiento eficiente en hardware acelerado.

El objetivo de este archivo es proporcionar una base robusta de pruebas automáticas que faciliten el desarrollo,
mantenimiento y refactorización del modelo Informer, asegurando su fiabilidad y facilitando la detección temprana
de errores o comportamientos inesperados.
"""
import torch
import pytest

from src.informer import Informer

# =======================
# Prueba de la salida del modelo
# =======================
def test_forward_shape():
    """
    Verifica que la salida del modelo Informer tenga la forma esperada.
    
    Este test crea un batch de datos sintéticos con un tamaño de batch, horizonte y tamaño de entrada definidos.
    Se construye el modelo y se le pasa el batch. Se espera que la salida tenga la forma (batch_size, h, 1),
    donde 'h' es el horizonte de predicción y '1' corresponde a la dimensión de salida por variable objetivo.
    """
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
    """
    Verifica que el modelo lance una excepción si el parámetro 'decoder_input_size_multiplier' es inválido.
    
    El valor permitido debe estar dentro de un rango específico. Si se pasa un valor fuera de ese rango,
    el modelo debe lanzar una excepción para evitar configuraciones erróneas.
    """
    with pytest.raises(Exception):
        Informer(h=2, input_size=4, decoder_input_size_multiplier=1.2)

def test_invalid_activation():
    """
    Verifica que el modelo lance una excepción si se especifica una función de activación no soportada.
    
    El modelo solo acepta ciertas funciones de activación. Si se pasa una cadena inválida, debe lanzar una excepción.
    """
    with pytest.raises(Exception):
        Informer(h=2, input_size=4, activation="tanh")

# ===========================
# Pruebas adicionales sugeridas
# ===========================

def test_backward_pass():
    """
    Comprueba que el modelo permite el cálculo de gradientes (backpropagation).
    
    Se crea un batch de entrada con requires_grad=True para la serie temporal. Se realiza un forward pass,
    se calcula una pérdida simple (media de la salida) y se ejecuta el backward pass. El test verifica que
    el gradiente de la entrada no sea None, lo que indica que el modelo es compatible con entrenamiento supervisado.
    """
    model = Informer(h=2, input_size=4)
    insample_y = torch.randn(2, 4, 1, requires_grad=True)
    futr_exog = torch.empty(2, 6, 0)
    batch = {"insample_y": insample_y, "futr_exog": futr_exog}
    output = model(batch)
    loss = output.mean()
    loss.backward()
    assert insample_y.grad is not None

def test_with_exogenous_inputs():
    """
    Verifica que el modelo procese correctamente variables exógenas.
    
    Se crea un batch con variables exógenas (futr_exog) y se comprueba que la salida tenga la forma esperada.
    Esto asegura que el modelo puede integrar información adicional en la predicción.
    """
    model = Informer(h=2, input_size=4)
    insample_y = torch.randn(2, 4, 1)
    futr_exog = torch.randn(2, 6, 3)  # 3 variables exógenas
    batch = {"insample_y": insample_y, "futr_exog": futr_exog}
    output = model(batch)
    assert output.shape == (2, 2, 1)

@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_various_batch_sizes(batch_size):
    """
    Prueba la flexibilidad del modelo ante diferentes tamaños de batch.
    
    Se parametriza el test para varios tamaños de batch y se verifica que la salida del modelo
    tenga la forma correcta en cada caso. Esto es importante para asegurar que el modelo
    puede ser usado en producción con diferentes configuraciones de batch.
    """
    model = Informer(h=2, input_size=4)
    insample_y = torch.randn(batch_size, 4, 1)
    futr_exog = torch.empty(batch_size, 6, 0)
    batch = {"insample_y": insample_y, "futr_exog": futr_exog}
    output = model(batch)
    assert output.shape == (batch_size, 2, 1)

def test_missing_keys_in_batch():
    """
    Verifica que el modelo lance un KeyError si falta una clave obligatoria en el batch de entrada.
    
    En este caso, se omite 'futr_exog' y se espera que el modelo lo detecte y lance la excepción correspondiente.
    """
    model = Informer(h=2, input_size=4)
    insample_y = torch.randn(2, 4, 1)
    with pytest.raises(KeyError):
        batch = {"insample_y": insample_y}  # Falta 'futr_exog'
        model(batch)

def test_cuda_support():
    """
    Comprueba que el modelo y los datos pueden ser movidos a GPU (CUDA) y que la salida también reside en la GPU.
    
    Solo se ejecuta si CUDA está disponible. Esto es importante para asegurar compatibilidad con entrenamiento acelerado por hardware.
    """
    if torch.cuda.is_available():
        model = Informer(h=2, input_size=4).cuda()
        insample_y = torch.randn(2, 4, 1, device="cuda")
        futr_exog = torch.empty(2, 6, 0, device="cuda")
        batch = {"insample_y": insample_y, "futr_exog": futr_exog}
        output = model(batch)
        assert output.is_cuda

def test_nan_input():
    """
    Verifica que el modelo lance una excepción si la entrada contiene valores NaN.
    
    Se introduce un NaN en la serie temporal de entrada y se espera que el modelo lo detecte y lance una excepción,
    previniendo así resultados inesperados o silenciosos durante el entrenamiento o inferencia.
    """
    model = Informer(h=2, input_size=4)
    insample_y = torch.randn(2, 4, 1)
    insample_y[0, 0, 0] = float('nan')
    futr_exog = torch.empty(2, 6, 0)
    batch = {"insample_y": insample_y, "futr_exog": futr_exog}
    with pytest.raises(Exception):
        model(batch)
