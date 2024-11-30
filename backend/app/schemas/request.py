from app.schemas.base_schema import ResponseModel
from typing import Optional

# Request data model 정의
class SimulationRequest(ResponseModel):
    simulation_id: str              # 시뮬레이션 고유 ID
    limestone_ratio: float          # 석회석 비율 (0~1)
    ggbs_ratio: float               # 고로슬래스 비율 (0~1)
    co2_emission_limestone: float   # 석회석으로 인한 CO2 배출량(kg)
    co2_emission_ggbs: float        # 고로슬래그로 인한 CO2 배출량 (kg)
    total_co2_emission: float       # 총 CO2 배출량 (kg)
    energy_consumption_kwh: float   # 총 에너지 소비량 (kWh)
    total_cost: float               # 총 전기 요금 (KRW)
    user_weight: float              # 사용자가 입력한 시멘트 기준 (1~100톤)
    compressive_strength: float     # 압축 강도

class SimulationParamsRequest(ResponseModel):
    simulation_id: str                                          # 시뮬레이션 고유 ID
    user_input_limestone_ratio: float                           # 석회석 비율 (0-1)
    user_input_ggbs_ratio: float                                # 고로슬래그 비율 (0-1)
    environmental_factors_temperature: Optional[float] = None   # 온도
    environmental_factors_pressure: Optional[float] = None      # 압력
    user_weight: float                                          