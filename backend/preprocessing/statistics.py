"""Basic image statistics — measurement only.

These are plain descriptive measurements of an uploaded image (mean, spread,
range and a coarse luminance histogram). They are **not** the illumination,
noise or quality analysis the research still has to define: no threshold, no
classification, no decision is made from them here.

Implemented with Pillow's ``ImageStat`` so no numerical library is required.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageStat

HISTOGRAM_BINS = 16


@dataclass(frozen=True)
class ChannelStatistics:
    """Descriptive statistics for one channel, on the 0-255 scale."""

    mean: float
    min: int
    max: int
    stddev: float


@dataclass(frozen=True)
class ImageStatistics:
    """Measurements that are safe to expose before any algorithm is chosen."""

    luminance: ChannelStatistics
    red: ChannelStatistics
    green: ChannelStatistics
    blue: ChannelStatistics
    luminance_histogram: tuple[int, ...]
    histogram_bins: int = HISTOGRAM_BINS

    @property
    def pixel_count(self) -> int:
        return sum(self.luminance_histogram)


def _channel(stat: ImageStat.Stat, index: int) -> ChannelStatistics:
    return ChannelStatistics(
        mean=round(stat.mean[index], 3),
        min=int(stat.extrema[index][0]),
        max=int(stat.extrema[index][1]),
        stddev=round(stat.stddev[index], 3),
    )


def _histogram(image: Image.Image, bins: int = HISTOGRAM_BINS) -> tuple[int, ...]:
    counts = [0] * bins
    bin_width = 256 / bins
    for value, count in enumerate(image.convert("L").histogram()):
        counts[min(int(value / bin_width), bins - 1)] += count
    return tuple(counts)


def compute_statistics(image: Image.Image) -> ImageStatistics:
    """Measure an RGB image. The input image is not modified."""
    rgb = image if image.mode == "RGB" else image.convert("RGB")

    return ImageStatistics(
        luminance=_channel(ImageStat.Stat(rgb.convert("L")), 0),
        red=_channel(ImageStat.Stat(rgb), 0),
        green=_channel(ImageStat.Stat(rgb), 1),
        blue=_channel(ImageStat.Stat(rgb), 2),
        luminance_histogram=_histogram(rgb),
    )
