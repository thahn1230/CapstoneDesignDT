from app.schemas.base_schema import ResponseModel
from typing import Optional
from pydantic import Field

# Request data model 정의

class UserInput(ResponseModel):
    limestone_ratio: float = Field(..., ge=0.0, le=1.0, description="사용자 입력 석회 비율 (0~1)")
    ggbs_ratio: float = Field(..., ge=0.0, le=1.0, description="사용자 입력 고로슬래그 비율 (0~1)")


class EnvironmentalFactors(ResponseModel):
    temperature: Optional[float] = Field(None, description="환경 요소 - 온도 (섭씨)", example=25.0)
    pressure: Optional[float] = Field(None, description="환경 요소 - 압력", example=101.3)


class SimulationRequest(ResponseModel):
    simulation_id: str = Field(..., description="시뮬레이션 고유 ID")
    user_input: UserInput
    environmental_factors: Optional[EnvironmentalFactors] = None
    user_weight: float = Field(..., ge=1.0, le=100.0, description="사용자 입력한 시멘트 기준 (1~100톤)")
    optimized_Select_1: int = Field(..., ge=0, le=3, description="사용자 최적화 요소 1 (0~3)")
    optimized_Select_2: int = Field(..., ge=0, le=3, description="사용자 최적화 요소 2 (0~3)")