from pydantic import BaseModel

class RecommendationRequest(BaseModel):
    cpu_request: float
    mem_request: float

    cpu_mean: float
    cpu_max: float
    cpu_std: float

    mem_mean: float
    mem_max: float
    mem_std: float


class RecommendationResponse(BaseModel):
    cpu_recommended: float
    mem_recommended: float
    decision_cpu: str
    decision_mem: str