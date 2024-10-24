from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas.request import SimulationRequest, SimulationParamsRequest
from app.schemas.response import SimulationResponse, SimulationResponseData

router = APIRouter()

@router.post("/update_simulation_data", response_model=SimulationResponse)
async def update_simulation_data(request: SimulationRequest):
    if request.limestone_ratio < 0 or request.limestone_ratio > 1:
        raise HTTPException(status_code=400, detail="limestone_ratio must be between 0 and 1.")

    if request.ggbs_ratio < 0 or request.ggbs_ratio > 1:
        raise HTTPException(status_code=400, detail="ggbs_ratio must be between 0 and 1.")

    # Unreal Engine에서 시뮬레이션 데이터를 처리하는 부분
    try:
        # 여기에 SimPy 연산 로직이 들어가야 하는것 같음
        updated_data = {
            "limestone_ratio": request.limestone_ratio,
            "ggbs_ratio": request.ggbs_ratio,
            "co2_emission_limestone": request.co2_emission_limestone,
            "co2_emission_ggbs": request.co2_emission_ggbs,
            "total_co2_emission": request.total_co2_emission,
            "energy_consumption_kwh": request.energy_consumption_kwh,
            "total_cost": request.total_cost
        }

        return SimulationResponse(
            status="success",
            message="Simulation updated successfully.",
            updated_simulation_data=SimulationResponseData(**updated_data)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Simulation update failed due to an internal error.")
    


@router.post("/update_simulation_params", response_model=SimulationResponse)
async def update_simulation_params(request: SimulationParamsRequest):
    # 필수 필드 검증
    if not (0 <= request.user_input_limestone_ratio <= 1):
        raise HTTPException(status_code=400, detail="limestone_ratio must be between 0 and 1.")
    
    if not (0 <= request.user_input_ggbs_ratio <= 1):
        raise HTTPException(status_code=400, detail="ggbs_ratio must be between 0 and 1.")

    # 환경 요소 처리 (optional 처리)
    temperature = request.environmental_factors_temperature if request.environmental_factors_temperature is not None else "default value"
    pressure = request.environmental_factors_pressure if request.environmental_factors_pressure is not None else "default value"
    
    # 시뮬레이션 처리 로직 (여기서는 단순히 데이터를 그대로 반환하는 구조)
    try:
        # 실제 처리 로직을 이곳에 구현 (예: Unreal Engine에서 받은 데이터를 SimPy로 전달 후 결과 처리)
        # 처리 성공 시 응답
        return SimulationResponse(
            status="success",
            message="Simulation updated successfully."
        )
    
    except Exception:
        # 처리 실패 시 응답
        raise HTTPException(status_code=500, detail="Simulation update failed due to an internal error.")
