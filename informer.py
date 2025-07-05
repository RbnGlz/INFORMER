"""
Compatibility module for backward compatibility.

This module re-exports the refactored components to maintain
compatibility with existing code that imports from informer.py.

For new code, please import directly from src.informer.
"""

import warnings

# Re-export main components from the new structure
try:
    from src.informer.models.informer import Informer
    from src.informer.models.components.attention import ProbAttention, ProbMask
    from src.informer.models.components.layers import ConvLayer
    
    # Maintain backward compatibility
    __all__ = ['Informer', 'ProbAttention', 'ProbMask', 'ConvLayer']
    
    # Issue deprecation warning
    warnings.warn(
        "Importing from 'informer.py' is deprecated. "
        "Please use 'from src.informer import Informer' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
except ImportError as e:
    raise ImportError(
        f"Could not import from new structure: {e}. "
        "Please ensure you have installed the package correctly. "
        "Try: pip install -e ."
    ) from e