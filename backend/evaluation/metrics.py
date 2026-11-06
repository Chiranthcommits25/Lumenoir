"""Evaluation structures and metric registry.

This module is the dependency-light face of the evaluation package: it can be
imported by the API without numpy, OpenCV, PyTorch, LPIPS or pyiqa. The actual
calculations live in ``reference_metrics`` / ``no_reference_metrics`` and are
reached through ``harness``, which imports those lazily.

Every metric below is implemented. Where a dependency is not installed, a
metric reports ``value=None`` with an explanatory note instead of a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from PIL import Image


class MetricStatus(StrEnum):
    """Implementation state of a metric."""

    IMPLEMENTED = "implemented"
    PLANNED = "planned"


class MetricName(StrEnum):
    """Metrics required by the rubric (Context §4 and §13, final_decision §6)."""

    PSNR = "psnr"
    SSIM = "ssim"
    LPIPS = "lpips"
    NIQE = "niqe"
    LATENCY = "latency"
    NOISE_SUPPRESSION = "noise_suppression"
    EDGE_PRESERVATION = "edge_preservation"
    COLOUR_DISTORTION = "colour_distortion"
    OVEREXPOSURE = "overexposure"
    HALO = "halo"
    LOE = "loe"
    BLACK_CRUSHING = "black_crushing"


@dataclass(frozen=True)
class MetricSpec:
    """Declares a metric, its requirements and its current status."""

    name: MetricName
    label: str
    status: MetricStatus
    requires_reference: bool
    requires_dependency: str | None
    description: str


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        MetricName.PSNR,
        "PSNR",
        MetricStatus.IMPLEMENTED,
        requires_reference=True,
        requires_dependency="scikit-image",
        description=(
            "Full-reference fidelity against the ground truth, with a GT-mean-aligned "
            "variant reported alongside the raw value."
        ),
    ),
    MetricSpec(
        MetricName.SSIM,
        "SSIM",
        MetricStatus.IMPLEMENTED,
        requires_reference=True,
        requires_dependency="scikit-image",
        description="Structural similarity against the ground truth, raw and mean-aligned.",
    ),
    MetricSpec(
        MetricName.LPIPS,
        "LPIPS",
        MetricStatus.IMPLEMENTED,
        requires_reference=True,
        requires_dependency="lpips + torch",
        description="Learned perceptual distance (AlexNet) against the reference pair.",
    ),
    MetricSpec(
        MetricName.NIQE,
        "NIQE",
        MetricStatus.IMPLEMENTED,
        requires_reference=False,
        requires_dependency="pyiqa + torch",
        description=(
            "No-reference perceptual quality on unseen images. Implementations differ "
            "from the MATLAB original, so use it for relative comparisons only."
        ),
    ),
    MetricSpec(
        MetricName.LATENCY,
        "Processing latency",
        MetricStatus.IMPLEMENTED,
        requires_reference=False,
        requires_dependency="psutil (peak RSS, optional)",
        description=(
            "Warm-up plus timed runs, median and P95, per-stage breakdown and peak RSS."
        ),
    ),
    MetricSpec(
        MetricName.NOISE_SUPPRESSION,
        "Noise suppression",
        MetricStatus.IMPLEMENTED,
        requires_reference=False,
        requires_dependency="numpy + opencv",
        description=(
            "Immerkaer sigma on flat regions divided by their mean luminance, plus a "
            "paired residual in flat regions."
        ),
    ),
    MetricSpec(
        MetricName.EDGE_PRESERVATION,
        "Edge preservation",
        MetricStatus.IMPLEMENTED,
        requires_reference=False,
        requires_dependency="numpy + opencv",
        description=(
            "Gradient-magnitude correlation on the strongest 10% of edges, plus local "
            "standard-deviation retention in textured patches."
        ),
    ),
    MetricSpec(
        MetricName.COLOUR_DISTORTION,
        "Colour distortion",
        MetricStatus.IMPLEMENTED,
        requires_reference=False,
        requires_dependency="scikit-image",
        description=(
            "CIEDE2000 against the reference when paired; mean wrapped hue difference "
            "on chroma > 10 pixels when unpaired."
        ),
    ),
    MetricSpec(
        MetricName.OVEREXPOSURE,
        "Overexposure",
        MetricStatus.IMPLEMENTED,
        requires_reference=False,
        requires_dependency="numpy",
        description=(
            "New clipped-pixel fraction at 0.98, plus the share of near-highlight "
            "input pixels pushed to clipping."
        ),
    ),
    MetricSpec(
        MetricName.HALO,
        "Halo analysis",
        MetricStatus.IMPLEMENTED,
        requires_reference=False,
        requires_dependency="numpy + opencv",
        description="Gradient-reversal rate on the strongest 5% of input edges.",
    ),
    MetricSpec(
        MetricName.LOE,
        "LOE",
        MetricStatus.IMPLEMENTED,
        requires_reference=False,
        requires_dependency="numpy + opencv",
        description=(
            "Lightness Order Error on 50x50 max-RGB lightness maps; the fraction "
            "of pixel pairs whose relative lightness order changes."
        ),
    ),
    MetricSpec(
        MetricName.BLACK_CRUSHING,
        "Black-level crushing",
        MetricStatus.IMPLEMENTED,
        requires_reference=False,
        requires_dependency="numpy",
        description=(
            "New crushed-black fraction: output max-channel <= 0.02 minus input "
            "max-channel <= 0.02."
        ),
    ),
)


@dataclass(frozen=True)
class MetricResult:
    """One measured value. ``value`` is None when it could not be computed."""

    name: MetricName
    value: float | None
    status: MetricStatus
    unit: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class EvaluationReport:
    """Metrics for one image pair, or for one image plus latency measurements."""

    results: tuple[MetricResult, ...]
    reference_available: bool
    processing_time_ms: float | None = None

    def value(self, name: MetricName) -> float | None:
        for result in self.results:
            if result.name is name:
                return result.value
        return None


def metric_specs() -> tuple[MetricSpec, ...]:
    """Return the declared metrics in report order."""
    return METRIC_SPECS


#: Legacy metric name -> key in the harness registry.
HARNESS_KEYS: dict[MetricName, str] = {
    MetricName.PSNR: "psnr",
    MetricName.SSIM: "ssim",
    MetricName.LPIPS: "lpips",
    MetricName.NIQE: "niqe",
    MetricName.NOISE_SUPPRESSION: "noise_relative",
    MetricName.EDGE_PRESERVATION: "edge_preservation",
    MetricName.COLOUR_DISTORTION: "delta_e00",
    MetricName.OVEREXPOSURE: "new_clipping",
    MetricName.HALO: "halo_rate",
    MetricName.LOE: "loe",
    MetricName.BLACK_CRUSHING: "black_crushing",
}


def compute_metrics(
    *,
    image: Image.Image,
    reference: Image.Image | None = None,
    original: Image.Image | None = None,
    processing_time_ms: float | None = None,
) -> EvaluationReport:
    """Compute the declared metrics for one output image.

    The harness is imported lazily so importing this module (and therefore the
    API) never requires numpy, OpenCV or PyTorch. Unavailable dependencies and
    missing references are reported as ``value=None`` with a note.

    ``original`` is the low-light input. When it is supplied without a
    reference, colour distortion is measured as a hue shift against that input.
    """
    from .harness import evaluate_arrays
    from .images import to_float32_rgb

    output = to_float32_rgb(image, name="output")
    reference_array = (
        to_float32_rgb(reference, name="reference") if reference is not None else None
    )
    original_array = (
        to_float32_rgb(original, name="original") if original is not None else None
    )

    keys = list(HARNESS_KEYS.values())
    if reference_array is None:
        keys = [key for key in keys if key not in {"psnr", "ssim", "lpips", "delta_e00"}]
        if original_array is not None:
            keys.append("hue_shift_deg")

    values = evaluate_arrays(
        output,
        reference=reference_array,
        original=original_array,
        keys=keys,
    )

    results: list[MetricResult] = []
    for name in MetricName:
        if name is MetricName.LATENCY:
            results.append(
                MetricResult(
                    name=name,
                    value=processing_time_ms,
                    status=MetricStatus.IMPLEMENTED,
                    unit="ms",
                    notes=None
                    if processing_time_ms is not None
                    else "No latency measurement was supplied.",
                )
            )
            continue

        key = HARNESS_KEYS[name]
        if key not in values:
            if name is MetricName.COLOUR_DISTORTION and "hue_shift_deg" in values:
                value = values["hue_shift_deg"]
                results.append(
                    MetricResult(
                        name=name,
                        value=value.value,
                        status=MetricStatus.IMPLEMENTED,
                        unit="degrees",
                        notes="Unpaired: hue shift against the input image."
                        if value.notes is None
                        else value.notes,
                    )
                )
                continue
            results.append(
                MetricResult(
                    name=name,
                    value=None,
                    status=MetricStatus.IMPLEMENTED,
                    notes="A ground-truth reference is required.",
                )
            )
            continue

        metric_value = values[key]
        results.append(
            MetricResult(
                name=name,
                value=metric_value.value,
                status=MetricStatus.IMPLEMENTED,
                unit=metric_value.unit,
                notes=metric_value.notes,
            )
        )

    return EvaluationReport(
        results=tuple(results),
        reference_available=reference_array is not None,
        processing_time_ms=processing_time_ms,
    )
