# INFORMER: Modelo de Pronóstico de Series Temporales

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

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/INFORMER.git
   cd INFORMER
   ```
2. (Opcional) Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   # Windows
   venv\\Scripts\\activate
   # macOS/Linux
   source venv/bin/activate
   ```
3. Instala dependencias:
   ```bash
   pip install --upgrade pip
   pip install torch numpy pandas matplotlib pytest neuralforecast
   ```
4. Verifica la instalación:
   ```bash
   python -c "import neuralforecast; print('NeuralForecast versión:', neuralforecast.__version__)"
   ```

## Uso

Ejemplo de entrenamiento e inferencia:

```python
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import Informer

# 1. Preparación de datos
data = pd.read_csv('data/serie.csv', parse_dates=['ds'])
data = data.sort_values(['unique_id', 'ds'])

# 2. Configuración del modelo
model = Informer(
    input_size=1,
    output_size=1,
    seq_len=96,
    label_len=48,
    pred_len=24,
    e_layers=3,
    d_layers=2,
    d_model=256,
    n_heads=4,
    dropout=0.1,
    attn='prob',
    distil=True
)

# 3. Entrenamiento
gf = NeuralForecast(models=[model], freq='D', n_jobs=4)
gf.fit(data)

# 4. Predicción
forecast = gf.predict()
print(forecast.tail(10))
```

## Estructura del proyecto

```
INFORMER/
├── images/                # Diagramas e ilustraciones para el README
├── informer.py            # Implementación de la clase Informer
├── nbs/                   # Notebooks de ejemplo
├── neuralforecast/        # Componentes auxiliares del modelo
│   ├── common/
│   └── losses/
├── tests/                 # Pruebas unitarias (pytest)
├── requirements.txt       # Dependencias
└── readme.md              # Documentación principal
```

## Pruebas unitarias (pytest)

1. Instala pytest:
   ```bash
   pip install pytest
   ```
2. Ejecuta las pruebas:
   ```bash
   pytest --maxfail=1 --disable-warnings -q
   ```

Las pruebas cubren formas de salida, reproducibilidad con semilla fija y rendimiento en secuencias largas.

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

