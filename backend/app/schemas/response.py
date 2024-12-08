from app.schemas.base_schema import ResponseModel
from typing import Optional
from pydantic import Field


# Response data model 정의
class UpdatedSimulationData(ResponseModel):
    simulation_id: str = Field(..., description="시뮬레이션 고유 ID (yyyyMMdd_001)")
    limestone_ratio: float = Field(..., ge=0.0, le=1.0, description="석회석 비율 (0~1)")
    ggbs_ratio: float = Field(..., ge=0.0, le=1.0, description="고로슬래그 비율 (0~1)")
    co2_emission_limestone: float = Field(..., description="석회석으로 인한 CO2 배출량 (kg)")
    co2_emission_ggbs: float = Field(..., description="고로슬래그로 인한 CO2 배출량 (kg)")
    compressive_strength: float = Field(..., description="시뮬레이션 예측 압축 강도 (MPa)")
    energy_consumption_kwh: float = Field(..., description="시뮬레이션 총 에너지 소비량 (kWh)")
    energy_cost: float = Field(..., description="전력 요금 (KRW)")
    material_cost: float = Field(..., description="원자재 가격 기반 총 요금 (KRW)")
    optimized_limestone_ratio: float = Field(..., description="최적화된 석회석 비율")
    optimized_ggbs_ratio: float = Field(..., description="최적화된 고로슬래그 비율")
    optimized_total_co2_emission: float = Field(..., description="최적화된 탄소배출량")
    optimized_compressive_strength: float = Field(..., description="최적화된 예측강도")
    optimized_energy_consumption_kw: float = Field(..., description="최적화된 에너지배출량")
    optimized_energy_cost: float = Field(..., description="최적화된 전력요금")
    baseline_co2_emission: float = Field(..., description="기준 대비 탄소 배출량 변화율")
    baseline_compressive_strength: float = Field(..., description="기준 대비 압축 강도 변화율")
    baseline_energy_consumption: float = Field(..., description="기준 대비 에너지 소비량 변화율")
    baseline_energy_cost: float = Field(..., description="기준 대비 전력 요금 변화율")


class SimulationResponse(ResponseModel):
    status: str = Field(..., description="상태 신호 ('received' 또는 'error')")
    message: str = Field(..., description="결과 메시지 ('Simulation updated successfully.' 또는 오류 메시지)")
    updated_simulation_data: Optional[UpdatedSimulationData] = Field(None, description="업데이트된 시뮬레이션 데이터")