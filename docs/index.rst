Informer Documentation
======================

Welcome to the Informer documentation. Informer is a state-of-the-art time series forecasting model that combines efficiency and accuracy for very long prediction horizons.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   contributing

Key Features
------------

* **ProbSparse Attention**: Reduces complexity from O(L²) to O(L log L)
* **Progressive Distilling**: Eliminates redundancy and stabilizes training
* **Encoder-Decoder Architecture**: Bidirectional information flow
* **Domain Versatility**: Validated across multiple domains

Quick Start
-----------

.. code-block:: python

   from informer import Informer
   import torch

   # Create model
   model = Informer(h=24, input_size=96)

   # Prepare data
   batch = {
       "insample_y": torch.randn(32, 96, 1),
       "futr_exog": torch.randn(32, 120, 0)
   }

   # Generate forecast
   forecast = model(batch)

API Reference
=============

.. toctree::
   :maxdepth: 2

   api/models
   api/components

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`