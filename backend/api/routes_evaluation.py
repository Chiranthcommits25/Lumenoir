"""Evaluation endpoints.

The endpoint reports the implemented metric registry and its dependency
requirements. Metric computation itself runs through the offline harness.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..evaluation.metrics import MetricStatus, metric_specs
from .schemas import MetricSpecSchema, MetricsResponse

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="List the declared evaluation metrics and their status",
)
def list_metrics() -> MetricsResponse:
    specs = metric_specs()
    implemented = any(spec.status is MetricStatus.IMPLEMENTED for spec in specs)

    return MetricsResponse(
        status="available" if implemented else "planned",
        metrics=[
            MetricSpecSchema(
                name=spec.name.value,
                label=spec.label,
                status=spec.status.value,
                requires_reference=spec.requires_reference,
                requires_dependency=spec.requires_dependency,
                description=spec.description,
            )
            for spec in specs
        ],
    )
