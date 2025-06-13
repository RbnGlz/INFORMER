# Informer

This repository contains a PyTorch implementation of the **Informer** architecture for long sequence forecasting. The code is auto-generated from the notebook [nbs/models.informer.ipynb](nbs/models.informer.ipynb).

Informer alleviates the heavy computational cost of vanilla Transformers using three key ideas:

1. **ProbSparse self-attention** with $O(L\log L)$ time and memory complexity.
2. A **self-attention distilling process** that focuses on important elements of long sequences.
3. An **MLP multi-step decoder** to predict long horizons in a single forward pass.

A more detailed explanation is available in the class docstring inside [`informer.py`](informer.py).

## Requirements

- [neuralforecast](https://github.com/Nixtla/neuralforecast)
- torch
- numpy

Install them with `pip`:

```bash
pip install neuralforecast torch numpy
```

## Quick example

```python
from informer import Informer

# minimal initialization
model = Informer(
    h=24,            # forecast horizon
    input_size=168,  # history length
)
```

This class can then be used with the **neuralforecast** training utilities.

## Running tests

To execute the unit tests, install the test dependencies (`pytest` and `torch`) if they are not already installed and run:

```bash
pytest
```

This command will discover and run all tests located under the `tests/` directory.
