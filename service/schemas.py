from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ContainerMetaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    container_id: str = Field(..., min_length=1)
    machine_id: str = Field(..., min_length=1)
    app_du: str | None = None
    status: str = "started"
    cpu_request: float = Field(..., gt=0)
    cpu_limit: float = Field(..., gt=0)
    mem_size: float = Field(..., gt=0)


class UsagePointInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_stamp: int = Field(..., ge=0)
    cpu_util_percent: float = Field(..., ge=0, le=100)
    mem_util_percent: float = Field(..., ge=0, le=100)
    cpi: float | None = None
    mem_gps: float | None = None
    mpki: float = 0.0
    net_in: float = 0.0
    net_out: float = 0.0
    disk_io_percent: float = 0.0


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ContainerMetaInput
    usage: list[UsagePointInput] = Field(..., min_length=1)
    include_features: bool = True
    include_window_series: bool = False
