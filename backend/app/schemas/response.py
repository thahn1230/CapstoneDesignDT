from app.schemas.base_schema import ResponseModel
from typing import Optional

# Response data model 정의
class SimulationResponseData(ResponseModel):
    limestone_ratio: float
    ggbs_ratio: float
    co2_emission_limestone: float
    co2_emission_ggbs: float
    total_co2_emission: float
    energy_consumption_kwh: float
    total_cost: float

class SimulationResponse(ResponseModel):
    status: str
    message: str
    updated_simulation_data: Optional[SimulationResponseData] = None