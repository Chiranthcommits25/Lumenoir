"""Serializers shared by the image and enhancement routes."""

from __future__ import annotations

from ..preprocessing.statistics import ChannelStatistics as ChannelStatisticsData
from ..preprocessing.statistics import ImageStatistics as ImageStatisticsData
from ..preprocessing.validation import ImageMetadata
from .schemas import ChannelStatistics, ImageInfo, ImageStatistics


def channel_schema(channel: ChannelStatisticsData) -> ChannelStatistics:
    return ChannelStatistics(
        mean=channel.mean, min=channel.min, max=channel.max, stddev=channel.stddev
    )


def statistics_schema(statistics: ImageStatisticsData) -> ImageStatistics:
    return ImageStatistics(
        luminance=channel_schema(statistics.luminance),
        red=channel_schema(statistics.red),
        green=channel_schema(statistics.green),
        blue=channel_schema(statistics.blue),
        luminance_histogram=list(statistics.luminance_histogram),
        histogram_bins=statistics.histogram_bins,
    )


def image_info(
    metadata: ImageMetadata, statistics: ImageStatisticsData | None = None
) -> ImageInfo:
    return ImageInfo(
        filename=metadata.filename,
        content_type=metadata.content_type,
        format=metadata.format,
        width=metadata.width,
        height=metadata.height,
        size_bytes=metadata.size_bytes,
        colour_mode=metadata.colour_mode,
        megapixels=metadata.megapixels,
        aspect_ratio=metadata.aspect_ratio,
        statistics=statistics_schema(statistics) if statistics else None,
    )
