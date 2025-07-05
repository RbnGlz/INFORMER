# INFORMER: Advanced Time Series Forecasting / Pronóstico Avanzado de Series Temporales

[![CI](https://github.com/RbnGlz/informer/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/RbnGlz/informer/actions)
[![codecov](https://codecov.io/gh/RbnGlz/informer/branch/main/graph/badge.svg)](https://codecov.io/gh/RbnGlz/informer)
[![PyPI version](https://badge.fury.io/py/informer-forecasting.svg)](https://badge.fury.io/py/informer-forecasting)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](#english) | [Español](#español)

---

## English

### Overview

**INFORMER** is a state-of-the-art time series forecasting model that combines efficiency and accuracy for very long prediction horizons. Its design is optimized to process sequences of hundreds or even thousands of steps, dramatically reducing training time and memory consumption compared to standard Transformers.

### Key Features

- **ProbSparse Attention**: Dynamically selects the most relevant queries, reducing complexity from O(L²) to O(L log L)
- **Progressive Distilling**: Each encoder layer includes a compression step that eliminates redundancy and stabilizes training
- **Encoder-Decoder Architecture**: Bidirectional information flow combining historical data and future context
- **Domain Versatility**: Validated in energy, traffic, finance, sales, and meteorology applications

### Quick Start

```bash
pip install informer-forecasting
```

```python
from informer import Informer
import torch

# Create model
model = Informer(h=24, input_size=96)

# Prepare data
batch = {
    "insample_y": torch.randn(32, 96, 1),      # Historical data
    "futr_exog": torch.randn(32, 120, 0)      # Future exogenous variables
}

# Generate forecast
forecast = model(batch)
print(f"Forecast shape: {forecast.shape}")  # [32, 24, 1]
```

### Installation for Development

```bash
git clone https://github.com/RbnGlz/informer.git
cd informer
make install-dev
```

### Running Tests

```bash
make test           # Run all tests
make test-parallel  # Run tests in parallel
make lint          # Code quality checks
make security      # Security scans
```

---

## Español

### Descripción General

**INFORMER** es un modelo de pronóstico de series temporales de última generación que combina eficiencia y precisión en horizontes de predicción muy extensos. Su diseño está optimizado para procesar secuencias de cientos o incluso miles de pasos, reduciendo drásticamente el tiempo de entrenamiento y el consumo de memoria en comparación con los Transformers estándar.

## Características principales

- **Atención ProbSparse**: Selecciona dinámicamente las consultas más relevantes, reduciendo la complejidad de O(L²) a O(L log L).
- **Distilling progresivo**: Cada capa del encoder incluye un paso de compresión que elimina redundancias y estabiliza el entrenamiento.
- **Arquitectura encoder-decoder**: Flujo bidireccional de información que combina datos históricos y contexto futuro.
- **Versatilidad de dominios**: Validado en energía, tráfico, finanzas, ventas y meteorología.

## Arquitectura del modelo

La arquitectura de INFORMER se articula en tres componentes:

### 1. Atención ProbSparse

Para cada consulta \(q_n\), se evalúa la relevancia:

$$
M(q_n, K) = \max_j \langle q_n, k_j \rangle - \frac{1}{L} \sum_{j=1}^L \langle q_n, k_j \rangle
$$

Se construye un subconjunto \(Q_{reduce}\) de tamaño \(u = O(L \log L)\) con las consultas de mayor valor \(M\). Al predecir demanda energética, por ejemplo, el modelo prioriza automáticamente las horas punta de consumo. Esta selección reduce el coste de atención a O(L log L) y mejora la calidad de la predicción.

### 2. Encoder-Decoder con Distilling

- **Encoder** (\(e\_layers\) bloques): cada bloque ejecuta:

  1. Multi-Head ProbSparse Self-Attention.
  2. Layer Normalization.
  3. Distilling (convolución 1D + pooling).

- **Decoder** (\(d\_layers\) bloques): cada bloque incluye:

  1. Multi-Head Attention sobre la salida comprimida del encoder.
  2. Masked Multi-Head ProbSparse Self-Attention para generación autoregresiva.
  3. Capa fully connected para proyección final.

Este diseño garantiza que las representaciones intermedias sean cada vez más concisas, conservando las dependencias a largo plazo.

### 3. Pila del encoder (Figura 3)



La pila mostrada en la Figura 3 incluye:

1. **Convolución 1D + ELU** para captar patrones locales.
2. **Max-Pooling** que reduce la secuencia a la mitad.
3. **Bloque de Atención ProbSparse** que prioriza zonas críticas de la serie.
4. **Distilling progresivo** que comprime la representación.
5. **Estructura piramidal** mediante réplicas de la pila con resolución reducida.

## Instalación

### Instalación desde PyPI (Recomendada)
```bash
pip install informer-forecasting
```

### Instalación desde Código Fuente
1. Clona el repositorio:
   ```bash
   git clone https://github.com/RbnGlz/informer.git
   cd informer
   ```

2. (Opcional) Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Instala en modo desarrollo:
   ```bash
   make install-dev
   ```

4. Verifica la instalación:
   ```bash
   python -c "from src.informer import Informer; print('Informer instalado correctamente')"
   ```

## Uso

Ejemplo de entrenamiento e inferencia:

### Uso Básico

```python
from src.informer import Informer
import torch

# 1. Configuración del modelo
model = Informer(
    h=24,                    # Horizonte de pronóstico
    input_size=96,           # Tamaño de ventana de entrada
    hidden_size=256,         # Dimensión del modelo
    encoder_layers=3,        # Capas del encoder
    decoder_layers=2,        # Capas del decoder
    n_head=4,               # Cabezas de atención
    dropout=0.1,            # Dropout
    factor=3                # Factor ProbSparse
)

# 2. Preparación de datos
batch_size = 32
batch = {
    "insample_y": torch.randn(batch_size, 96, 1),     # Serie temporal histórica
    "futr_exog": torch.randn(batch_size, 120, 0)      # Variables exógenas futuras
}

# 3. Pronóstico
model.eval()
with torch.no_grad():
    forecast = model(batch)
    
print(f"Forma del pronóstico: {forecast.shape}")  # [32, 24, 1]
```

### Uso Avanzado con NeuralForecast

```python
import pandas as pd
from neuralforecast import NeuralForecast
from src.informer import Informer

# 1. Preparación de datos
data = pd.read_csv('data/serie.csv', parse_dates=['ds'])
data = data.sort_values(['unique_id', 'ds'])

# 2. Configuración del modelo
model = Informer(
    h=24,
    input_size=96,
    hidden_size=256,
    encoder_layers=3,
    decoder_layers=2,
    n_head=4,
    dropout=0.1,
    max_steps=1000
)

# 3. Entrenamiento
gf = NeuralForecast(models=[model], freq='D')
gf.fit(data)

# 4. Predicción
forecast = gf.predict()
print(forecast.tail(10))
```

## Estructura del Proyecto

```
informer/
├── src/                   # Código fuente
│   └── informer/
│       ├── models/        # Modelos y componentes
│       │   ├── informer.py           # Modelo principal
│       │   └── components/           # Componentes reutilizables
│       │       ├── attention.py      # Mecanismos de atención
│       │       └── layers.py         # Capas auxiliares
│       ├── utils/         # Utilidades
│       └── losses/        # Funciones de pérdida
├── tests/                 # Pruebas
│   ├── unit/             # Pruebas unitarias
│   ├── integration/      # Pruebas de integración
│   └── fixtures/         # Datos de prueba
├── docs/                  # Documentación
├── notebooks/             # Notebooks de ejemplo
├── requirements/          # Dependencias organizadas
│   ├── base.txt          # Dependencias base
│   └── dev.txt           # Dependencias de desarrollo
├── .github/               # Configuración CI/CD
│   └── workflows/
├── pyproject.toml         # Configuración del proyecto
├── Makefile              # Automatización de tareas
└── README.md             # Este archivo
```

### Herramientas de Desarrollo

- **Testing**: pytest con cobertura de código
- **Linting**: flake8, mypy para calidad de código
- **Formatting**: black, isort para formato consistente
- **Security**: bandit, safety para escaneo de seguridad
- **CI/CD**: GitHub Actions para integración continua
- **Documentation**: Sphinx para documentación API

## Testing y Calidad de Código

### Ejecutar Pruebas

```bash
# Todas las pruebas con cobertura
make test

# Pruebas en paralelo (más rápido)
make test-parallel

# Solo pruebas unitarias
pytest tests/unit/ -v

# Solo pruebas de integración
pytest tests/integration/ -v
```

### Herramientas de Calidad

```bash
# Verificar formato de código
make lint

# Formatear código automáticamente
make format

# Escaneos de seguridad
make security

# Todo junto
make all
```

### Cobertura de Pruebas

Las pruebas cubren:
- ✅ Formas de salida y tipos de datos
- ✅ Reproducibilidad con semilla fija
- ✅ Rendimiento en secuencias largas
- ✅ Manejo de errores y validación
- ✅ Compatibilidad con CUDA
- ✅ Flujo de gradientes en entrenamiento
- ✅ Integración end-to-end

## Casos de uso

- Demanda energética: predicción horaria/mensual de consumo.
- Tráfico urbano: estimación de flujos vehiculares en tiempo real.
- Análisis de ventas: detección de tendencias estacionales y promociones.
- Finanzas cuantitativas: pronóstico de precios y volatilidad.
- Meteorología: modelado de temperatura y precipitaciones.

### Ejemplo de variables exógenas

```python
data['day_of_week'] = data['ds'].dt.weekday
data['month'] = data['ds'].dt.month
```

Estos campos pueden incorporarse en `Informer` para mejorar la precisión.

## Contribuciones

1. Haz fork del repositorio.
2. Crea una rama con tu mejora.
3. Añade pruebas que validen tu cambio.
4. Abre un pull request describiendo los objetivos y resultados.

## Licencia

Este proyecto se distribuye bajo la licencia **MIT**. Revisa `LICENSE` para más detalles.

