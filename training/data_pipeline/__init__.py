"""Offline dataset preparation pipeline for LUMENOIR.

The pipeline is split into small modules so each stage is independently
testable:

    config        constants, dataset layouts and the exact split counts
    sources       official dataset sources, licences and presence checks
    organisation  discovery, low/high pairing and manifest records
    validation    corrupted / truncated / unsupported image detection
    hashing       64-bit perceptual hashes and an exact Hamming index
    leakage       duplicate and cross-dataset leakage detection
    preprocessing decoding, resizing, cropping and augmentation
    synthetic     the specified extreme-dark synthesis formula
    splits        deterministic split generation with leakage filtering
    statistics    dataset statistics and cleaning reports
    pipeline      orchestration used by the CLI

Raw images are never modified or deleted. Every generated artefact is written
under the processed / manifest / split / report directories.
"""

from __future__ import annotations

from .config import (
    APP_INFERENCE_MAX_EDGE,
    CROP_SIZE,
    DARKEN_PROBABILITY,
    HAMMING_THRESHOLD,
    PHASH_BITS,
    SEED,
    SICE_LONG_EDGE,
    SSIM_SIZE,
    SSIM_THRESHOLD,
    SYNTHETIC_LOL_SIZE,
)

__all__ = [
    "APP_INFERENCE_MAX_EDGE",
    "CROP_SIZE",
    "DARKEN_PROBABILITY",
    "HAMMING_THRESHOLD",
    "PHASH_BITS",
    "SEED",
    "SICE_LONG_EDGE",
    "SSIM_SIZE",
    "SSIM_THRESHOLD",
    "SYNTHETIC_LOL_SIZE",
]
