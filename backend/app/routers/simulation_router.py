import simpy

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas.request import SimulationRequest
from app.schemas.response import SimulationResponse, UpdatedSimulationData

# from app.utils.simulate.simulate_utils import simulate
# from app.utils.simulate.simulate1128 import simulate
from app.utils.simulate.simulate1206 import simulate

router = APIRouter()

@router.post("/update_simulation_params", response_model=SimulationResponse)
async def update_simulation_params(request: SimulationRequest):
    try:
        # SimPy 환경 생성
        simpy_env = simpy.Environment()

        # SimPy 프로세스 실행
        sim_process = simpy_env.process(
            simulate(
                simpy_env,
                request_status="received",
                simulation_id=request.simulation_id,
                limestone_ratio=request.user_input.limestone_ratio,
                ggbs_ratio=request.user_input.ggbs_ratio,
                user_weight=request.user_weight,
                optimized_select_1=request.optimized_Select_1,
                optimized_select_2=request.optimized_Select_2,
                before_co2_emission_limestone=0.0,
                before_co2_emission_ggbs=0.0,
                before_total_co2_emission=0.0,
                before_energy_consumption_kwh=0.0,
                before_total_cost=0.0,
            )
        )
        simpy_env.run()  # SimPy 시뮬레이션 실행

        # SimPy 프로세스에서 반환된 값 가져오기
        sim_result = sim_process.value  # simulate 함수의 반환값

        # 응답 생성
        updated_data = UpdatedSimulationData(**sim_result)
        return SimulationResponse(
            status="success",
            message="Simulation updated successfully.",
            updated_simulation_data=updated_data,
        )

    except Exception as e:
        # 예외 처리
        raise HTTPException(status_code=500, detail=f"Simulation update failed: {str(e)}")
