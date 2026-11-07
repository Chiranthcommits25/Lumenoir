"""Zero-DCE++ (NoirNet) inference wrapper.

The network itself lives in :mod:`backend.deep_learning.zero_dce_net`, which
imports PyTorch. This module stays importable without PyTorch or NumPy so the
API can start with the inference stack absent; both are imported in ``load()``.

Checkpoint verified on 6 November 2026:

- source: ``Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth`` in
  ``Li-Chongyi/Zero-DCE_extension`` (official Zero-DCE++ repository)
- size: 52,395 bytes, git blob ``3d1a7b8f5aee769d8ae55531cd43c4cf8d45efa8``
- SHA-256: ``ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84``
- 28 tensors, 10,561 parameters, strict key match with the architecture

Input/output:
- input is an RGB PIL image scaled to [0, 1]; no mean/std normalisation
- the image is padded to a multiple of ``scale_factor`` (the authors' script
  crops instead, discarding up to ``scale_factor - 1`` border pixels)
- output is RGB [0, 1] clipped for 8-bit encoding, at the exact input size
- the curve map is exposed as both the raw tanh output and the documented
  ``alpha`` (``-raw``), because ``x + raw*(x^2 - x) == x + (-raw)*x*(1-x)``
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ..inference.base import EnhancementModel
from ..inference.device import resolve_device
from ..inference.errors import (
    CheckpointError,
    InferenceFailedError,
    TorchNotInstalledError,
)
from ..inference.registry import register_model

logger = logging.getLogger(__name__)

MODEL_ID = "zero_dce_plus_plus"
DISPLAY_NAME = "Zero-DCE++ (NoirNet)"

DEFAULT_SCALE_FACTOR = 12
SCALE_FACTOR_ENV = "LUMENOIR_ZERODCE_SCALE_FACTOR"
MIN_SCALE_FACTOR = 1
MAX_SCALE_FACTOR = 12

MAX_GAIN = 12.0
HIGHLIGHT_START = 0.92

EXPECTED_PARAMETERS = 10_561
EXPECTED_TENSORS = 28
EXPECTED_CHECKPOINT_BYTES = 52_395

_WARM_UP_SIZE = 24


@dataclass(frozen=True)
class PreparedInput:
    """Tensor plus the geometry needed to undo the padding."""

    tensor: Any
    original_size: tuple[int, int]
    padded_size: tuple[int, int]


@dataclass(frozen=True)
class TensorModelOutput:
    """Device tensors returned by :meth:`infer_tensor`."""

    image: Any
    curve_alpha: Any


@dataclass(frozen=True)
class RawOutput:
    """Network output before it is converted back to an image."""

    enhanced: Any
    raw_alpha: Any
    curve_alpha: Any
    original_size: tuple[int, int]
    padded_size: tuple[int, int]
    original: Any


def _resolve_scale_factor(explicit: int | None) -> int:
    if explicit is None:
        raw = os.getenv(SCALE_FACTOR_ENV, "").strip()
        value = DEFAULT_SCALE_FACTOR if not raw else int(raw)
    else:
        value = int(explicit)

    if not MIN_SCALE_FACTOR <= value <= MAX_SCALE_FACTOR:
        raise ValueError(
            f"{SCALE_FACTOR_ENV} must be in [{MIN_SCALE_FACTOR}, {MAX_SCALE_FACTOR}], "
            f"got {value}."
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@register_model(MODEL_ID)
class ZeroDCEPlusPlusModel(EnhancementModel):
    """Load the pretrained Zero-DCE++ checkpoint and run the curve network."""

    model_id = MODEL_ID
    display_name = DISPLAY_NAME

    def __init__(
        self,
        weights_path: Path,
        *,
        device: str = "cpu",
        scale_factor: int | None = None,
    ) -> None:
        super().__init__(weights_path, device=device)
        self.scale_factor = _resolve_scale_factor(scale_factor)
        self.parameter_count: int | None = None
        self._torch: Any | None = None
        self._numpy: Any | None = None
        self._net: Any | None = None
        self._precision = "float32"

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Load the checkpoint once, strictly, and warm the network up."""
        if self._loaded and self._net is not None:
            return

        torch = self._import_torch()
        numpy = self._import_numpy()
        self.device = resolve_device(self.device)

        from .zero_dce_net import ZeroDCEPlusPlusNet

        net = ZeroDCEPlusPlusNet(scale_factor=self.scale_factor)
        state = self._read_state_dict(torch)
        self._validate_against(net, state)

        try:
            net.load_state_dict(state, strict=True)
        except Exception as exc:  # pragma: no cover - validate_against runs first
            raise CheckpointError(f"Could not load the checkpoint: {exc}") from exc

        try:
            net.to(self.device).eval()
            dtype = torch.float16 if self._precision == "float16" else torch.float32
            net.to(dtype=dtype)
        except Exception as exc:
            raise InferenceFailedError(
                f"Could not move Zero-DCE++ to device '{self.device}': {exc}"
            ) from exc

        self._torch = torch
        self._numpy = numpy
        self._net = net
        self.parameter_count = sum(parameter.numel() for parameter in net.parameters())
        self._warm_up()
        self._loaded = True

    def close(self) -> None:
        self._net = None
        self._loaded = False
        torch = self._torch
        self._torch = None
        self._numpy = None
        if torch is not None and self.device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("MPS cache cleanup failed", exc_info=True)

    def set_precision(self, precision: str) -> None:
        """Select float32 or float16 inference for the loaded network."""
        name = (precision or "float32").strip().lower()
        aliases = {
            "fp32": "float32",
            "float": "float32",
            "fp16": "float16",
            "half": "float16",
        }
        name = aliases.get(name, name)
        if name not in {"float32", "float16"}:
            raise ValueError(
                f"Unsupported inference precision {precision!r}; "
                "use 'float32' or 'float16'."
            )
        if name == "float16" and self.device == "cpu":
            raise RuntimeError(
                "float16 inference is not supported on the CPU device."
            )
        self._precision = name
        if self._net is not None:
            self._require_torch()
            self._net.to(dtype=self._torch_dtype())

    def _torch_dtype(self):
        torch = self._require_torch()
        return torch.float16 if self._precision == "float16" else torch.float32

    # -- model stages ------------------------------------------------------

    def infer_tensor(self, tensor):
        """Run Zero-DCE++ on an HxWx3 RGB tensor and return device tensors."""
        torch = self._require_torch()
        net = self._require_net()
        if not hasattr(tensor, "shape") or tensor.ndim != 3 or tensor.shape[2] != 3:
            raise InferenceFailedError(
                "Zero-DCE++ expected an HxWx3 tensor for tensor inference."
            )
        if not tensor.is_floating_point():
            raise InferenceFailedError(
                "Zero-DCE++ tensor input must use a floating-point dtype."
            )

        work = tensor.to(device=self.device, dtype=self._torch_dtype())
        height, width = work.shape[:2]
        pad_bottom = (-height) % self.scale_factor
        pad_right = (-width) % self.scale_factor
        chw = work.permute(2, 0, 1).unsqueeze(0)
        if pad_bottom or pad_right:
            mode = (
                "reflect"
                if pad_bottom < height and pad_right < width
                else "replicate"
            )
            chw = torch.nn.functional.pad(
                chw, (0, pad_right, 0, pad_bottom), mode=mode
            )

        try:
            with torch.inference_mode():
                enhanced, raw_alpha = net(chw)
            original = work[:, :, :height, :width].permute(1, 2, 0).float().contiguous()
            enhanced = enhanced[0, :, :height, :width].clamp(0.0, 1.0).permute(1, 2, 0).float().contiguous()
            gain = (enhanced / original.clamp(min=1e-4)).clamp(min=1.0, max=MAX_GAIN)
            enhanced = (original * gain).clamp(0.0, 1.0)
            lum = (0.2126 * original[..., 0] + 0.7152 * original[..., 1] + 0.0722 * original[..., 2]).clamp(0.0, 1.0)
            highlight_weight = ((lum - HIGHLIGHT_START) / (1.0 - HIGHLIGHT_START)).clamp(0.0, 1.0)
            enhanced = enhanced * (1.0 - highlight_weight.unsqueeze(-1)) + original * highlight_weight.unsqueeze(-1)
            enhanced = enhanced.clamp(0.0, 1.0).contiguous()
            image = enhanced
            alpha = (
                (-raw_alpha)[0, :, :height, :width]
                .permute(1, 2, 0)
                .float()
                .contiguous()
            )
        except Exception as exc:
            raise InferenceFailedError(
                f"Zero-DCE++ tensor inference failed: {exc}"
            ) from exc

        if tuple(image.shape) != (height, width, 3):
            raise InferenceFailedError(
                f"Zero-DCE++ returned {tuple(image.shape)}, expected "
                f"{(height, width, 3)}."
            )
        return TensorModelOutput(image=image, curve_alpha=alpha)

    def preprocess(self, image: Image.Image) -> PreparedInput:
        torch = self._require_torch()
        numpy = self._require_numpy()
        if self.scale_factor < 1:  # pragma: no cover - validated in __init__
            raise InferenceFailedError("scale_factor must be positive.")

        rgb = numpy.asarray(image.convert("RGB"), dtype=numpy.float32) / 255.0
        height, width = rgb.shape[:2]
        pad_bottom = (-height) % self.scale_factor
        pad_right = (-width) % self.scale_factor

        tensor = torch.from_numpy(
            numpy.ascontiguousarray(rgb.transpose(2, 0, 1))
        ).unsqueeze(0)

        if pad_bottom or pad_right:
            # Reflect needs padding smaller than the dimension; tiny images use
            # edge replication instead of failing.
            mode = "reflect" if pad_bottom < height and pad_right < width else "replicate"
            tensor = torch.nn.functional.pad(
                tensor, (0, pad_right, 0, pad_bottom), mode=mode
            )

        return PreparedInput(
            tensor=tensor.to(self.device),
            original_size=(width, height),
            padded_size=(width + pad_right, height + pad_bottom),
        )

    def infer(self, prepared: PreparedInput) -> RawOutput:
        torch = self._require_torch()
        net = self._require_net()

        with torch.inference_mode():
            enhanced, raw_alpha = net(
                prepared.tensor.to(dtype=self._torch_dtype())
            )

        return RawOutput(
            enhanced=enhanced,
            raw_alpha=raw_alpha,
            curve_alpha=-raw_alpha,
            original_size=prepared.original_size,
            padded_size=prepared.padded_size,
            original=prepared.tensor,
        )

    def postprocess(self, output: RawOutput, target_size: tuple[int, int]) -> Image.Image:
        if not isinstance(output, RawOutput):
            raise InferenceFailedError(
                "Zero-DCE++ expected its own RawOutput from infer()."
            )

        torch = self._require_torch()
        numpy = self._require_numpy()
        width, height = output.original_size

        try:
            import numpy as np

            original = output.original[0, :, :height, :width].permute(1, 2, 0).float().cpu().numpy()
            enhanced = output.enhanced[0, :, :height, :width].clamp(0.0, 1.0).permute(1, 2, 0).float().cpu().numpy()
            gain = np.clip(enhanced / np.clip(original, 1e-4, None), 1.0, MAX_GAIN)
            enhanced = np.clip(original * gain, 0.0, 1.0)
            lum = 0.2126 * original[..., 0] + 0.7152 * original[..., 1] + 0.0722 * original[..., 2]
            highlight_weight = np.clip((lum - HIGHLIGHT_START) / (1.0 - HIGHLIGHT_START), 0.0, 1.0)[..., None]
            enhanced = enhanced * (1.0 - highlight_weight) + original * highlight_weight
            enhanced = np.clip(enhanced, 0.0, 1.0)
            array = np.round(enhanced * 255.0).astype("uint8")
        except Exception as exc:
            raise InferenceFailedError(
                f"Could not convert the Zero-DCE++ output to an image: {exc}"
            ) from exc

        if array.shape[:2] != (height, width):
            raise InferenceFailedError(
                f"Zero-DCE++ output has shape {array.shape[:2]}, expected {(height, width)}."
            )

        result = Image.fromarray(numpy.ascontiguousarray(array))
        if result.size != tuple(target_size):
            raise InferenceFailedError(
                f"Zero-DCE++ output is {result.size}, expected {tuple(target_size)}."
            )
        return result

    # -- checkpoint helpers ------------------------------------------------

    def _read_state_dict(self, torch: Any) -> dict[str, Any]:
        if not self.weights_path.is_file():
            raise CheckpointError(f"Checkpoint '{self.weights_path}' does not exist.")

        self._log_fingerprint()

        try:
            payload = torch.load(
                self.weights_path, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise CheckpointError(
                f"Could not read checkpoint '{self.weights_path}': {exc}"
            ) from exc

        if (
            isinstance(payload, dict)
            and "state_dict" in payload
            and isinstance(payload["state_dict"], dict)
        ):
            payload = payload["state_dict"]

        if not isinstance(payload, dict):
            raise CheckpointError(
                "Checkpoint does not contain a state dict; expected a mapping of "
                "tensor names to tensors."
            )

        state: dict[str, Any] = {}
        for key, value in payload.items():
            if not torch.is_tensor(value):
                raise CheckpointError(
                    f"Checkpoint entry '{key}' is not a tensor; refusing to load it."
                )
            state[key.removeprefix("module.")] = value
        return state

    def _validate_against(self, net: Any, state: dict[str, Any]) -> None:
        total = sum(int(value.numel()) for value in state.values())
        if total != EXPECTED_PARAMETERS:
            hint = ""
            if total > 50_000:
                hint = (
                    " This matches the original Zero-DCE checkpoint; NoirNet uses "
                    "Zero-DCE++ (about 10K parameters)."
                )
            raise CheckpointError(
                f"Checkpoint has {total:,} parameters, expected "
                f"{EXPECTED_PARAMETERS:,}.{hint}"
            )

        if len(state) != EXPECTED_TENSORS:
            raise CheckpointError(
                f"Checkpoint has {len(state)} tensors, expected {EXPECTED_TENSORS}."
            )

        expected = set(net.state_dict())
        actual = set(state)
        if expected != actual:
            missing = sorted(expected - actual)[:5]
            unexpected = sorted(actual - expected)[:5]
            raise CheckpointError(
                "Checkpoint does not match the Zero-DCE++ architecture. "
                f"Missing keys: {missing}. Unexpected keys: {unexpected}."
            )

    def _log_fingerprint(self) -> None:
        size = self.weights_path.stat().st_size
        logger.info(
            "Zero-DCE++ checkpoint %s: %d bytes, sha256=%s",
            self.weights_path,
            size,
            _sha256(self.weights_path),
        )
        if size != EXPECTED_CHECKPOINT_BYTES:
            logger.warning(
                "Zero-DCE++ checkpoint size is %d bytes; the published pretrained "
                "checkpoint is %d bytes. Continuing because the state dict matches "
                "the architecture (a fine-tuned checkpoint will differ).",
                size,
                EXPECTED_CHECKPOINT_BYTES,
            )

    def _warm_up(self) -> None:
        torch = self._require_torch()
        net = self._require_net()
        try:
            with torch.inference_mode():
                sample = torch.zeros(
                    1,
                    3,
                    _WARM_UP_SIZE,
                    _WARM_UP_SIZE,
                    device=self.device,
                    dtype=self._torch_dtype(),
                )
                net(sample)
        except Exception as exc:
            self._net = None
            raise InferenceFailedError(
                f"Zero-DCE++ warm-up inference failed on device '{self.device}': {exc}"
            ) from exc

    # -- dependency helpers ------------------------------------------------

    def _import_torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise TorchNotInstalledError() from exc
        return torch

    def _import_numpy(self) -> Any:
        try:
            import numpy
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise InferenceFailedError(
                "NumPy is required for Zero-DCE++ inference. "
                "Install it with: pip install -r requirements-inference.txt"
            ) from exc
        return numpy

    def _require_torch(self) -> Any:
        if self._torch is None:
            raise InferenceFailedError("Zero-DCE++ is not loaded; call load() first.")
        return self._torch

    def _require_numpy(self) -> Any:
        if self._numpy is None:
            raise InferenceFailedError("Zero-DCE++ is not loaded; call load() first.")
        return self._numpy

    def _require_net(self) -> Any:
        if self._net is None:
            raise InferenceFailedError("Zero-DCE++ is not loaded; call load() first.")
        return self._net
