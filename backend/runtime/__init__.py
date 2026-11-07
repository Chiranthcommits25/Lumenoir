"""Device-aware runtime support for the enhancement pipeline.

The package is importable without PyTorch. CPU-only deployments never import
it beyond :mod:`backend.runtime.config`.
"""

from .config import PipelineRuntime, resolve_runtime

__all__ = ["PipelineRuntime", "resolve_runtime"]
