from fastapi import APIRouter
from app.api.schemas import RecommendationRequest, RecommendationResponse
from app.services.recommendation_service import recommend_cpu, recommend_memory, decision

router = APIRouter()

@router.post("/recommendation", response_model=RecommendationResponse)
def get_recommendation(req: RecommendationRequest):

    cpu_rec = recommend_cpu(req.cpu_mean, req.cpu_std)
    mem_rec = recommend_memory(req.mem_max)

    return RecommendationResponse(
        cpu_recommended=cpu_rec,
        mem_recommended=mem_rec,
        decision_cpu=decision(req.cpu_request, cpu_rec),
        decision_mem=decision(req.mem_request, mem_rec),
    )