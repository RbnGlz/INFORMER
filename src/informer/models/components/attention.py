"""Componentes de atención para el modelo Informer."""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


class ProbMask:
    """
    Máscara probabilística para la atención Informer.
    
    Esta clase implementa la máscara utilizada en el mecanismo de atención
    ProbSparse para manejar secuencias causales y no causales.
    """
    
    def __init__(
        self, 
        B: int, 
        H: int, 
        L: int, 
        index: torch.Tensor, 
        scores: torch.Tensor, 
        device: str = "cpu"
    ) -> None:
        """
        Inicializa la máscara probabilística.
        
        Args:
            B: Tamaño del batch
            H: Número de cabezas de atención
            L: Longitud de la secuencia
            index: Índices seleccionados para la atención sparse
            scores: Puntuaciones de atención
            device: Dispositivo de cómputo ('cpu' o 'cuda')
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], 
            torch.arange(H)[None, :, None], 
            index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self) -> torch.Tensor:
        """Retorna la máscara computada."""
        return self._mask


class ProbAttention(nn.Module):
    """
    Mecanismo de atención ProbSparse del modelo Informer.
    
    Implementa atención eficiente con complejidad O(L log L) en lugar de O(L²)
    mediante el muestreo inteligente de consultas relevantes.
    
    Características principales:
    - Selección probabilística de consultas top-k
    - Reducción significativa de complejidad computacional
    - Mantenimiento de la calidad de atención para secuencias largas
    """
    
    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ) -> None:
        """
        Inicializa el módulo de atención probabilística.
        
        Args:
            mask_flag: Si aplicar máscara causal para autoregresión
            factor: Factor de sampling para ProbSparse (afecta complejidad)
            scale: Factor de escala para las puntuaciones (por defecto: 1/sqrt(d))
            attention_dropout: Probabilidad de dropout en la atención
            output_attention: Si retornar los pesos de atención para visualización
        """
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        sample_k: int, 
        n_top: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula la atención probabilística Q-K.
        
        Args:
            Q: Tensor de consultas [B, H, L, D]
            K: Tensor de claves [B, H, L, D]
            sample_k: Número de claves a muestrear
            n_top: Número top de consultas a seleccionar
            
        Returns:
            Tuple con puntuaciones Q-K y índices top
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Calcular Q_K muestreado
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # Encontrar las consultas Top_k con medida de sparsity
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Usar las Q reducidas para calcular Q_K
        Q_reduce = Q[
            torch.arange(B)[:, None, None], 
            torch.arange(H)[None, :, None], 
            M_top, :
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V: torch.Tensor, L_Q: int) -> torch.Tensor:
        """
        Obtiene el contexto inicial para la atención.
        
        Args:
            V: Tensor de valores [B, H, L_V, D]
            L_Q: Longitud de las consultas
            
        Returns:
            Contexto inicial
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # usar máscara
            assert L_Q == L_V  # requiere L_Q == L_V para self-attention
            context = V.cumsum(dim=-2)
        return context

    def _update_context(
        self, 
        context_in: torch.Tensor, 
        V: torch.Tensor, 
        scores: torch.Tensor, 
        index: torch.Tensor, 
        L_Q: int, 
        attn_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Actualiza el contexto con las consultas seleccionadas.
        
        Args:
            context_in: Contexto de entrada
            V: Tensor de valores
            scores: Puntuaciones de atención
            index: Índices seleccionados
            L_Q: Longitud de consultas
            attn_mask: Máscara de atención externa
            
        Returns:
            Tuple con contexto actualizado y pesos de atención opcionales
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            prob_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(prob_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[
            torch.arange(B)[:, None, None], 
            torch.arange(H)[None, :, None], 
            index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V], device=attn.device) / L_V).type_as(attn)
            attns[
                torch.arange(B)[:, None, None], 
                torch.arange(H)[None, :, None], 
                index, :
            ] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        attn_mask: Optional[torch.Tensor],
        tau: Optional[torch.Tensor] = None, 
        delta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Pase hacia adelante del mecanismo de atención probabilística.
        
        Args:
            queries: Tensor de consultas [B, L_Q, H, D]
            keys: Tensor de claves [B, L_K, H, D]
            values: Tensor de valores [B, L_V, H, D]
            attn_mask: Máscara de atención opcional
            tau: Parámetro temporal (no usado actualmente)
            delta: Parámetro delta (no usado actualmente)
            
        Returns:
            Tuple con contexto resultante y pesos de atención opcionales
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype("int").item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype("int").item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Agregar factor de escala
        scale = self.scale or 1.0 / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
            
        # Obtener el contexto
        context = self._get_initial_context(values, L_Q)
        
        # Actualizar el contexto con las consultas top_k seleccionadas
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.contiguous(), attn