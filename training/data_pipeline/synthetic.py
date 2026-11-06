"""The specified extreme-dark synthesis formula.

The formula is copied from ``final_decision.md`` §1.5 without modification:

    lin   = img ** 2.2
    s     = 10 ** uniform(log10(0.03), log10(0.5))
    x     = lin * s
    sig_s = 10 ** uniform(-3, -2)
    sig_r = 10 ** uniform(-3, log10(1.5e-2))
    n     = normal(0, 1) * sqrt(sig_s * x + sig_r**2)
    out   = clip(x + n, 0, 1) ** (1 / 2.2)
    out   = round(out * 255) / 255

Linear-domain darkening with shot and read noise, then back to sRGB and 8-bit
quantisation. The augmentation is applied during fine-tuning with probability
0.5; the offline generator exists for the fixed synthetic evaluation sets.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

from .config import SEED


class SyntheticDarkError(ValueError):
    """Raised for inputs the formula is not defined on."""


def _as_float_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise SyntheticDarkError(
            f"Expected an HxWx3 RGB image, got shape {array.shape}."
        )
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise SyntheticDarkError("Image contains NaN or infinite values.")
    if array.min() < 0.0 or array.max() > 1.0:
        raise SyntheticDarkError("Float images must lie in [0, 1].")
    return np.ascontiguousarray(array)


def synth_dark(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Darken ``image`` in the linear domain and add shot/read noise.

    Args:
        image: HxWx3 RGB image, float32 in [0, 1] (or uint8).
        rng: NumPy generator, so the result is reproducible from a seed.

    Returns:
        A float32 HxWx3 image in [0, 1], quantised to 8 bits.
    """
    img = _as_float_image(image)

    lin = img**2.2
    scale = 10 ** rng.uniform(np.log10(0.03), np.log10(0.5))
    x = lin * scale
    shot_coefficient = 10 ** rng.uniform(-3, -2)
    read_coefficient = 10 ** rng.uniform(-3, np.log10(1.5e-2))
    noise = rng.normal(0, 1, x.shape) * np.sqrt(
        shot_coefficient * x + read_coefficient**2
    )
    out = np.clip(x + noise, 0, 1) ** (1 / 2.2)
    return (np.round(out * 255) / 255).astype(np.float32)


def generate_synthetic_darks(
    sources: Sequence[Path] | Iterable[Path],
    output_dir: Path,
    *,
    count: int,
    seed: int = SEED,
    overwrite: bool = False,
) -> list[Path]:
    """Write ``count`` synthetic dark images derived from ``sources``.

    ``count`` must be given explicitly; the pipeline never guesses how many
    synthetic images a split needs. Sources are sampled deterministically and
    each output gets its own generator seeded from ``(seed, index)``.
    """
    source_list = sorted(Path(path) for path in sources)
    if not source_list:
        raise SyntheticDarkError("No source images were provided.")
    if count <= 0:
        raise SyntheticDarkError("count must be a positive integer.")

    output_dir.mkdir(parents=True, exist_ok=True)
    selection = np.random.default_rng(seed).choice(
        len(source_list), size=count, replace=count > len(source_list)
    )

    written: list[Path] = []
    for index, source_index in enumerate(selection):
        source = source_list[int(source_index)]
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise SyntheticDarkError(f"Could not read source image '{source}'.")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        rng = np.random.default_rng(seed + index)
        darkened = synth_dark(rgb, rng)

        target = output_dir / f"{index:04d}_{source.stem}.png"
        written.append(target)
        if target.exists() and not overwrite:
            continue
        save_png(darkened, target)

    return written


def save_png(image: np.ndarray, path: Path) -> Path:
    """Write an RGB float32 image as an 8-bit PNG without colour conversion."""
    array = _as_float_image(image)
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    uint8 = np.clip(np.round(bgr * 255.0), 0, 255).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", uint8)
    if not ok:  # pragma: no cover - encoding a valid array cannot fail
        raise SyntheticDarkError(f"Could not encode PNG for '{path}'.")
    buffer.tofile(str(path))
    return path
