"""Capas y componentes auxiliares para el modelo Informer."""

import torch
import torch.nn as nn
from typing import Tuple


class ConvLayer(nn.Module):
    """
    Capa convolucional para la compresión progresiva del encoder.
    
    Implementa una capa de convolución 1D seguida de normalización, activación
    y max pooling para reducir la longitud de secuencia a la mitad mientras
    preserva las características importantes.
    
    Esta capa es fundamental para el proceso de destilación progresiva que
    caracteriza al modelo Informer.
    """

    def __init__(self, c_in: int) -> None:
        """
        Inicializa la capa convolucional.
        
        Args:
            c_in: Número de canales de entrada (dimensión del modelo)
        """
        super().__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pase hacia adelante de la capa convolucional.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, channels]
            
        Returns:
            Tensor de salida con longitud de secuencia reducida a la mitad
            [batch_size, seq_len//2, channels]
        """
        # Transponer para formato conv1d: [batch, channels, seq_len]
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        # Transponer de vuelta: [batch, seq_len, channels]
        x = x.transpose(1, 2)
        return x