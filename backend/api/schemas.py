"""Pydantic schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    service: str
    version: str
    enhancement_pipeline: str = Field(
        description="Implementation status of the enhancement pipeline.",
        examples=["implemented"],
    )
    model_state: str = Field(
        default="not_selected",
        description="Lifecycle state of the configured inference model.",
    )
    model_ready: bool = False


class ChannelStatistics(BaseModel):
    """Descriptive statistics for one channel, 0-255 scale."""

    mean: float
    min: int
    max: int
    stddev: float


class ImageStatistics(BaseModel):
    """Measurements of the uploaded image. Not a quality assessment."""

    luminance: ChannelStatistics
    red: ChannelStatistics
    green: ChannelStatistics
    blue: ChannelStatistics
    luminance_histogram: list[int]
    histogram_bins: int


class ImageInfo(BaseModel):
    """Basic information about a validated uploaded image."""

    filename: str
    content_type: str
    format: str = Field(description="Detected image format (JPEG or PNG).")
    width: int
    height: int
    size_bytes: int
    colour_mode: str = Field(description="Pillow colour mode, e.g. RGB or RGBA.")
    megapixels: float
    aspect_ratio: float
    statistics: ImageStatistics | None = Field(
        default=None,
        description="Descriptive measurements; null when statistics are not requested.",
    )


class ImageInfoResponse(BaseModel):
    ok: bool = True
    message: str = "Image validated successfully."
    image: ImageInfo


class DimensionLimits(BaseModel):
    min_width: int
    min_height: int
    max_width: int
    max_height: int
    max_megapixels: float


class UploadLimits(BaseModel):
    """Server-side upload constraints, so the UI can show the real limits."""

    max_size_bytes: int
    allowed_extensions: list[str]
    allowed_mime_types: list[str]
    dimensions: DimensionLimits


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    ok: bool = False
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Pipeline / enhancement
# ---------------------------------------------------------------------------


class ModelStatusSchema(BaseModel):
    state: str = Field(examples=["not_selected", "weights_missing", "ready"])
    model_id: str
    device: str
    weights_path: str | None = None
    message: str
    ready: bool
    load_duration_ms: float | None = None


class InferenceSettingsSchema(BaseModel):
    device: str
    precision: str = "float32"
    max_edge: int
    max_inference_pixels: int


class StageCapabilitySchema(BaseModel):
    name: str
    status: str = Field(description="implemented or not_implemented")
    description: str


class ModeCapabilitySchema(BaseModel):
    mode: str
    label: str
    available: bool
    required_stages: list[str]


class ArtifactSchema(BaseModel):
    """One planned intermediate output, available only when actually produced."""

    kind: str
    label: str
    description: str
    produced_by: str | None = None
    available: bool
    applicable: bool = True
    reason: str | None = None
    width: int | None = None
    height: int | None = None
    encoding: str | None = Field(
        default=None, description="'base64' when data is present."
    )
    media_type: str | None = None
    data: str | None = Field(
        default=None, description="Base64 image payload; null while unavailable."
    )


class StageReportSchema(BaseModel):
    name: str
    status: str
    duration_ms: float | None = None
    detail: str | None = None


class PipelineStatusResponse(BaseModel):
    ready: bool = Field(description="True only if some mode can produce an output.")
    stages: list[StageCapabilitySchema]
    modes: list[ModeCapabilitySchema]
    model: ModelStatusSchema
    inference: InferenceSettingsSchema
    artifacts: list[ArtifactSchema]


class PipelineRunSchema(BaseModel):
    mode: str
    processing_status: str = Field(examples=["unavailable", "completed", "failed"])
    processing_time_ms: float
    timings_ms: dict[str, float]
    stages: list[StageReportSchema]
    artifacts: list[ArtifactSchema]


class EnhancementOutputSchema(BaseModel):
    encoding: str = "base64"
    media_type: str = "image/png"
    filename: str
    data: str


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class MetricSpecSchema(BaseModel):
    name: str
    label: str
    status: str
    requires_reference: bool
    requires_dependency: str | None = None
    description: str


class MetricsResponse(BaseModel):
    status: str = Field(
        description="available when the metric registry is implemented.",
        examples=["available"],
    )
    metrics: list[MetricSpecSchema]
