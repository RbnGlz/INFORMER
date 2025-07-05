"""
Modelo Informer para pronósticos de series temporales de largo horizonte.

Esta implementación optimizada del modelo Informer incluye las tres características
distintivas: atención ProbSparse, destilación progresiva y decoder multi-paso.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from neuralforecast.common._modules import (
    TransEncoderLayer,
    TransEncoder,
    TransDecoderLayer,
    TransDecoder,
    DataEmbedding,
    AttentionLayer,
)
from neuralforecast.common._base_model import BaseModel
from neuralforecast.losses.pytorch import MAE

from .components.attention import ProbAttention
from .components.layers import ConvLayer


class Informer(BaseModel):
    """
    Modelo Informer para pronósticos de series temporales.

    El modelo Informer aborda los desafíos de complejidad computacional del 
    Transformer vanilla para pronósticos de largo horizonte. La arquitectura 
    tiene tres características distintivas:
    
    1) Un mecanismo de auto-atención ProbSparse con complejidad O(L log L) 
       en tiempo y memoria.
    2) Un proceso de destilación de auto-atención que prioriza la atención 
       y maneja eficientemente secuencias de entrada largas.
    3) Un decoder MLP multi-paso que predice secuencias largas de series 
       temporales en una sola operación forward en lugar de paso a paso.

    El modelo utiliza un enfoque de tres componentes para definir su embedding:
    1) Emplea características autoregresivas codificadas obtenidas de una red convolucional.
    2) Usa embeddings posicionales relativos a ventana derivados de funciones armónicas.
    3) Se utilizan embeddings posicionales absolutos obtenidos de características de calendario.

    Referencias:
        Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, 
        Hui Xiong, Wancai Zhang. "Informer: Beyond Efficient Transformer 
        for Long Sequence Time-Series Forecasting" (https://arxiv.org/abs/2012.07436)
    """

    # Atributos de clase
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(
        self,
        h: int,
        input_size: int,
        futr_exog_list: Optional[list] = None,
        hist_exog_list: Optional[list] = None,
        stat_exog_list: Optional[list] = None,
        exclude_insample_y: bool = False,
        decoder_input_size_multiplier: float = 0.5,
        hidden_size: int = 128,
        dropout: float = 0.05,
        factor: int = 3,
        n_head: int = 4,
        conv_hidden_size: int = 32,
        activation: str = "gelu",
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        distil: bool = True,
        loss: nn.Module = MAE(),
        valid_loss: Optional[nn.Module] = None,
        max_steps: int = 5000,
        learning_rate: float = 1e-4,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled: bool = False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        optimizer: Optional[Any] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[Any] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        **trainer_kwargs: Any,
    ) -> None:
        """
        Inicializa el modelo Informer.

        Args:
            h: Horizonte de pronóstico.
            input_size: Longitud máxima de secuencia para backpropagation truncada.
            futr_exog_list: Lista de columnas exógenas futuras.
            hist_exog_list: Lista de columnas exógenas históricas.
            stat_exog_list: Lista de columnas exógenas estáticas.
            exclude_insample_y: Si True, el modelo omite características autoregresivas.
            decoder_input_size_multiplier: Multiplicador del tamaño de entrada del decoder.
            hidden_size: Unidades de embeddings y encoders.
            dropout: Dropout a través de la arquitectura Informer.
            factor: Factor de atención Probsparse.
            n_head: Controla el número de cabezas de multi-head attention.
            conv_hidden_size: Canales del encoder convolucional.
            activation: Activación, opciones: ['relu', 'gelu'].
            encoder_layers: Número de capas para el encoder.
            decoder_layers: Número de capas para el decoder.
            distil: Si el decoder Informer usa bottlenecks.
            loss: Módulo de pérdida para entrenamiento.
            valid_loss: Módulo de pérdida para validación.
            max_steps: Número máximo de pasos de entrenamiento.
            learning_rate: Tasa de aprendizaje.
            num_lr_decays: Número de decaimientos de learning rate.
            early_stop_patience_steps: Pasos de paciencia para early stopping.
            val_check_steps: Pasos entre verificaciones de pérdida de validación.
            batch_size: Número de series diferentes en cada batch.
            valid_batch_size: Tamaño de batch para validación y test.
            windows_batch_size: Número de ventanas a muestrear en cada batch de entrenamiento.
            inference_windows_batch_size: Número de ventanas para inferencia.
            start_padding_enabled: Si hacer padding al inicio con zeros.
            step_size: Tamaño de paso entre ventanas de datos temporales.
            scaler_type: Tipo de scaler para normalización de entradas temporales.
            random_seed: Semilla aleatoria para inicializadores.
            drop_last_loader: Si descartar el último batch no completo.
            alias: Nombre personalizado del modelo.
            optimizer: Optimizador especificado por el usuario.
            optimizer_kwargs: Parámetros del optimizador.
            lr_scheduler: Programador de learning rate.
            lr_scheduler_kwargs: Parámetros del programador de LR.
            dataloader_kwargs: Parámetros del DataLoader.
            **trainer_kwargs: Argumentos heredados del trainer de PyTorch Lightning.
        """
        super().__init__(
            h=h,
            input_size=input_size,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            futr_exog_list=futr_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            drop_last_loader=drop_last_loader,
            alias=alias,
            random_seed=random_seed,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )

        # Validación de arquitectura
        self.label_len = int(np.ceil(input_size * decoder_input_size_multiplier))
        if (self.label_len >= input_size) or (self.label_len <= 0):
            raise ValueError(
                f"decoder_input_size_multiplier={decoder_input_size_multiplier} "
                f"debe estar en el rango (0,1). Resultó en label_len={self.label_len} "
                f"con input_size={input_size}"
            )

        if activation not in ["relu", "gelu"]:
            raise ValueError(
                f"activation='{activation}' no soportada. "
                f"Opciones válidas: ['relu', 'gelu']"
            )

        # Configuración del modelo
        self.c_out = getattr(self.loss, 'outputsize_multiplier', 1)
        self.output_attention = False
        self.enc_in = 1
        self.dec_in = 1

        # Embeddings
        self.enc_embedding = DataEmbedding(
            c_in=self.enc_in,
            exog_input_size=self.futr_exog_size,
            hidden_size=hidden_size,
            pos_embedding=True,
            dropout=dropout,
        )
        self.dec_embedding = DataEmbedding(
            self.dec_in,
            exog_input_size=self.futr_exog_size,
            hidden_size=hidden_size,
            pos_embedding=True,
            dropout=dropout,
        )

        # Encoder con atención ProbSparse y destilación progresiva
        self.encoder = TransEncoder(
            [
                TransEncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=self.output_attention,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size,
                    conv_hidden_size,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(encoder_layers)
            ],
            (
                [ConvLayer(hidden_size) for l in range(encoder_layers - 1)]
                if distil
                else None
            ),
            norm_layer=torch.nn.LayerNorm(hidden_size),
        )
        
        # Decoder con atención ProbSparse
        self.decoder = TransDecoder(
            [
                TransDecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    AttentionLayer(
                        ProbAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size,
                    conv_hidden_size,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(decoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_size),
            projection=nn.Linear(hidden_size, self.c_out, bias=True),
        )

    def forward(self, windows_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Pase hacia adelante del modelo Informer.

        Args:
            windows_batch: Diccionario con 'insample_y' y 'futr_exog'
                - insample_y: Tensor [batch_size, input_size, 1] con datos históricos
                - futr_exog: Tensor [batch_size, input_size + h, n_exog] con variables exógenas

        Returns:
            Tensor de pronósticos [batch_size, h, output_size]
        """
        # Parsear el batch de ventanas
        insample_y = windows_batch["insample_y"]
        futr_exog = windows_batch["futr_exog"]

        # Preparar marcas temporales para encoder y decoder
        if self.futr_exog_size > 0:
            x_mark_enc = futr_exog[:, : self.input_size, :]
            x_mark_dec = futr_exog[:, -(self.label_len + self.h) :, :]
        else:
            x_mark_enc = None
            x_mark_dec = None

        # Preparar entrada del decoder con zeros para posiciones futuras
        x_dec = torch.zeros(
            size=(len(insample_y), self.h, 1), 
            device=insample_y.device,
            dtype=insample_y.dtype
        )
        x_dec = torch.cat([insample_y[:, -self.label_len :, :], x_dec], dim=1)

        # Encoder: procesamiento de la secuencia histórica
        enc_out = self.enc_embedding(insample_y, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Decoder: generación del pronóstico
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # Extraer solo las predicciones futuras
        forecast = dec_out[:, -self.h :]
        
        return forecast