"""Deep learning models.

Importing this package registers every available architecture with
``backend.inference.registry``. The model modules do not import PyTorch or NumPy
at module level, so this package is safe to import in a deployment that has not
installed the inference dependencies.

Implemented: Zero-DCE++ (NoirNet). The enhancement pipeline stage that calls it
is still pending; see ``backend/enhancement/stages.py``.
"""

from __future__ import annotations

from .zero_dce_plus_plus import MODEL_ID, ZeroDCEPlusPlusModel

__all__ = ["MODEL_ID", "ZeroDCEPlusPlusModel"]
