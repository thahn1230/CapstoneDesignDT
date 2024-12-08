import simpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
import joblib
import json
from skopt import BayesSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import os
import math

# 가격정의
limestone_price_per_ton = 8168
ggbs_price_per_ton = 22160
energy_price_per_kwh = 120

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model1206.joblib')
model = joblib.load(model_path)

csv_path = os.path.join(current_dir, '1ton_total_simulation.csv')
data_df = pd.read_csv(csv_path)

def interpolate_data(limestone_ratio):
    # Limestone 비율에 따른 보간을 수행
    limestone_keys = data_df['Limestone_Ratio']
    # 각 열에 대해 보간 수행
    interpolated_values = {}
    for column in data_df.columns[2:]:
        y = data_df[column]
        interpolated_values[column] = np.interp(limestone_ratio, limestone_keys, y)

    return interpolated_values

interpolated_result = interpolate_data(0.5)
optimal_co2_emission = interpolated_result['Total_CO2_Emission(kg)'] * 1000

# 최적화 지표에 따른 가중치 계산 함수
def calculate_optimization_score(ggbs_ratio, optimized_select_1, optimized_select_2, co2_emission, compressive_strength, energy_consumption, energy_cost):
    # 순서대로 co2, 강도, energy comsumption, energy cost 가중치 (default)
    weights = [0.5,-0.5,0.5,0.5]

    # 선택한 가중치는 2배로
    weights[optimized_select_1]=weights[optimized_select_1]*2
    weights[optimized_select_2]=weights[optimized_select_2]*2

    # 항목별 점수 100점만점으로 환산
    co2_emission = ((375.212508-co2_emission)/(375.212508 - 85.27557)) * 100
    compressive_strength = ((90-compressive_strength)/(90-20)) * 100
    energy_consumption = ((0.508829851-energy_consumption)/(0.508829851 - 0.115643148)) * 100
    energy_cost = ((0.508829851*120-energy_cost)/(0.508829851*120 - 0.115643148*120)) * 100

    # 최적화 점수 계산
    score = (co2_emission * weights[0]) + (compressive_strength * weights[1]) + (energy_consumption * weights[2]) + (energy_cost * weights[3])
    return score

# objective 함수 정의
def objective(ggbs_ratio, optimized_select_1, optimized_select_2):
    interpolated_result = interpolate_data(1 - ggbs_ratio)
    total_co2_emission = interpolated_result['Total_CO2_Emission(kg)']
    energy_consumption = interpolated_result['Energy_Consumption(kWh)']
    energy_cost = energy_consumption * energy_price_per_kwh
    compressive_strength = model.predict(pd.DataFrame({'Cement': [550 * (1 - ggbs_ratio)], 'Fly Ash': [80], 'Water': [220], 'Age': [28], 'Total Binder': [550], 'Water-Binder Ratio': [0.4], 'Binder Ratio': [ggbs_ratio]}))[0]
    
    score = calculate_optimization_score(ggbs_ratio, optimized_select_1, optimized_select_2, total_co2_emission, compressive_strength, energy_consumption, energy_cost)
    
    return -score  # 최소화하기 위해 음수로 반환

def bayesian_optimization(optimized_select_1, optimized_select_2):
    # 베이지안 최적화 진행
    print("good")
    result = gp_minimize(lambda ggbs_ratio: objective(ggbs_ratio[0], optimized_select_1, optimized_select_2), [(0.0, 1.0)], n_calls=30, random_state=42)
    return result.x[0] 

# 최적화된 GGBS 비율을 시각화하는 함수
def plot_optimization_graph(optimized_select_1, optimized_select_2):
    ggbs_ratios = np.linspace(0, 1, 100)
    scores = []
    for ggbs_ratio in ggbs_ratios:
        score = objective(ggbs_ratio, optimized_select_1, optimized_select_2)
        scores.append(-score)
    
    # 그래프 그리기
    plt.plot(ggbs_ratios, scores, label='Optimization Score')
    plt.xlabel('GGBS Ratio')
    plt.ylabel('Optimization Score')
    plt.title('Optimal GGBS Ratio vs. Optimization Score')
    plt.grid(True)
    plt.show()

def simulate(env, **kwargs):
    yield env.timeout(0)  # SimPy 이벤트 처리

    ##### 입력 변수 #####
    limestone_ratio = kwargs.get('limestone_ratio')
    ggbs_ratio = kwargs.get('ggbs_ratio')
    user_weight = kwargs.get('user_weight')
    optimized_select_1 = kwargs.get('optimized_select_1')
    optimized_select_2 = kwargs.get('optimized_select_2')
    before_total_co2_emission = kwargs.get('before_total_co2_emission')
    before_energy_consumption_kwh = kwargs.get('before_energy_consumption_kwh')
    before_total_cost = kwargs.get('before_total_cost')

    ##### 계산 #####
    interpolated_result = interpolate_data(limestone_ratio)
    co2_emission_limestone = interpolated_result['CO2_Emission_Limestone(kg)'] * user_weight
    co2_emission_ggbs = interpolated_result['CO2_Emission_GGBS(kg)'] * user_weight
    total_co2_emission = co2_emission_limestone + co2_emission_ggbs
    energy_consumption_kwh = interpolated_result['Energy_Consumption(kWh)'] * user_weight
    energy_cost = energy_consumption_kwh * energy_price_per_kwh
    material_cost = (limestone_ratio * limestone_price_per_ton / 1000 * user_weight) + (
        ggbs_ratio * ggbs_price_per_ton / 1000 * user_weight
    )

    # 안전한 계산: 분모가 0인지 확인
    if before_total_co2_emission != 0:
        baseline_co2_emission = (
            (before_total_co2_emission - total_co2_emission) / before_total_co2_emission
        ) * 100
    else:
        baseline_co2_emission = 0.0

    if before_energy_consumption_kwh != 0:
        baseline_energy_consumption = (
            (before_energy_consumption_kwh - energy_consumption_kwh) / before_energy_consumption_kwh
        ) * 100
    else:
        baseline_energy_consumption = 0.0

    if before_total_cost != 0:
        baseline_energy_cost = ((before_total_cost - energy_cost) / before_total_cost) * 100
    else:
        baseline_energy_cost = 0.0

    # 압축 강도 계산
    new_input = pd.DataFrame({
        'Cement': [550 * (1 - ggbs_ratio)],
        'Fly Ash': [80],
        'Water': [220],
        'Age': [28],
        'Total Binder': [550],
        'Water-Binder Ratio': [0.4],
        'Binder Ratio': [ggbs_ratio],
    })
    compressive_strength = model.predict(new_input)[0]

    # 최적화
    optimized_ggbs_ratio = bayesian_optimization(optimized_select_1, optimized_select_2)
    optimized_limestone_ratio = 1 - optimized_ggbs_ratio
    optimized_interpolated_result = interpolate_data(optimized_limestone_ratio)
    optimized_total_co2_emission = (
        optimized_interpolated_result['Total_CO2_Emission(kg)'] * user_weight
    )
    optimized_energy_consumption_kwh = (
        optimized_interpolated_result['Energy_Consumption(kWh)'] * user_weight
    )
    optimized_energy_cost = optimized_energy_consumption_kwh * energy_price_per_kwh

    optimized_input = pd.DataFrame({
        'Cement': [550 * (1 - optimized_ggbs_ratio)],
        'Fly Ash': [80],
        'Water': [220],
        'Age': [28],
        'Total Binder': [550],
        'Water-Binder Ratio': [0.4],
        'Binder Ratio': [optimized_ggbs_ratio],
    })
    optimized_compressive_strength = model.predict(optimized_input)[0]

    ##### 유효성 검증: inf 또는 NaN 값 처리 #####
    def safe_value(value):
        return value if math.isfinite(value) else 0.0

    return {
        "simulation_id": kwargs.get('simulation_id'),
        "limestone_ratio": limestone_ratio,
        "ggbs_ratio": ggbs_ratio,
        "co2_emission_limestone": safe_value(co2_emission_limestone),
        "co2_emission_ggbs": safe_value(co2_emission_ggbs),
        "total_co2_emission": safe_value(total_co2_emission),
        "energy_consumption_kwh": safe_value(energy_consumption_kwh),
        "energy_cost": safe_value(energy_cost),
        "material_cost": safe_value(material_cost),
        "baseline_co2_emission": safe_value(baseline_co2_emission),
        "baseline_energy_consumption": safe_value(baseline_energy_consumption),
        "baseline_energy_cost": safe_value(baseline_energy_cost),
        "compressive_strength": safe_value(compressive_strength),
        "optimized_limestone_ratio": safe_value(optimized_limestone_ratio),
        "optimized_ggbs_ratio": safe_value(optimized_ggbs_ratio),
        "optimized_total_co2_emission": safe_value(optimized_total_co2_emission),
        "optimized_compressive_strength": safe_value(optimized_compressive_strength),
        "optimized_energy_consumption_kw": safe_value(optimized_energy_consumption_kwh),
        "optimized_energy_cost": safe_value(optimized_energy_cost),
        "baseline_compressive_strength": safe_value(baseline_energy_consumption),  # 임시 데이터
    }



# def simulate(env, **kwargs):

#     # SimPy에서 시간 흐름 처리
#     yield env.timeout(0) 

#     ##### 언리얼에서 받은 변수들 #####
#     request_status = kwargs.get('request_status')
#     simulation_id = kwargs.get('simulation_id')
#     limestone_ratio = kwargs.get('limestone_ratio')
#     ggbs_ratio = kwargs.get('ggbs_ratio')
#     user_weight = kwargs.get('user_weight')
#     optimized_select_1 = kwargs.get('optimized_select_1')
#     optimized_select_2 = kwargs.get('optimized_select_2')
#     before_co2_emission_limestone = kwargs.get('before_co2_emission_limestone')
#     before_co2_emission_ggbs = kwargs.get('before_co2_emission_ggbs')
#     before_total_co2_emission = kwargs.get('before_total_co2_emission')
#     before_energy_consumption_kwh = kwargs.get('before_energy_consumption_kwh')
#     before_total_cost = kwargs.get('before_total_cost')
    
  
#     ##### 언리얼로 보내야하는 변수들 ##### 
#     co2_emission_limestone = 0.0
#     co2_emission_ggbs = 0.0
#     compressive_strength = 0.0
#     energy_consumption_kwh = 0.0
#     energy_cost = 0.0
#     material_cost = 0.0
#     optimized_limestone_ratio = 0.0
#     optimized_ggbs_ratio = 0.0
#     optimized_total_co2_emission = 0.0
#     optimized_compressive_strength = 0.0
#     optimized_energy_consumption_kwh = 0.0
#     optimized_energy_cost = 0.0
#     baseline_co2_emission = 0.0
#     baseline_compressive_strength = 0.0
#     baseline_energy_consumption = 0.0
#     baseline_energy_cost = 0.0
#     status = "success"
#     message = "successfully"

#     # co2_emission_limestone, co2_emission_ggbs, energy_consumption_kwh, energy_cost, material_cost
#     # baseline_co2_emission, baseline_energy_consumption, baseline_energy_cost

#     interpolated_result = interpolate_data(limestone_ratio)
#     co2_emission_limestone = interpolated_result['CO2_Emission_Limestone(kg)'] * user_weight
#     co2_emission_ggbs = interpolated_result['CO2_Emission_GGBS(kg)'] * user_weight
#     total_co2_emission = co2_emission_limestone + co2_emission_ggbs
#     energy_consumption_kwh = interpolated_result['Energy_Consumption(kWh)'] * user_weight
#     energy_cost = energy_consumption_kwh * energy_price_per_kwh
#     material_cost = (limestone_ratio * limestone_price_per_ton/1000 * user_weight) + (ggbs_ratio * ggbs_price_per_ton/1000 * user_weight)
    
#     baseline_co2_emission = (before_total_co2_emission-total_co2_emission)/before_total_co2_emission * 100
#     baseline_energy_consumption = (before_energy_consumption_kwh-energy_consumption_kwh)/before_energy_consumption_kwh * 100
#     baseline_energy_cost = (before_total_cost-energy_cost)/before_total_cost * 100


#     # compressive_strength, baseline_compressive_strength

#     new_input = pd.DataFrame({
#     'Cement': [550 * (1 - ggbs_ratio)],
#     'Fly Ash': [80],
#     'Water': [220],
#     'Age': [28],
#     'Total Binder': [550],
#     'Water-Binder Ratio': [0.4],
#     'Binder Ratio': [ggbs_ratio],
#     })

#     compressive_strength = model.predict(new_input)[0]
#     baseline_compressive_strength = ((baseline_compressive_strength-compressive_strength)/baseline_compressive_strength) * 100

#     co2_emission_limestone = interpolated_result['CO2_Emission_Limestone(kg)'] * user_weight
#     co2_emission_ggbs = interpolated_result['CO2_Emission_GGBS(kg)'] * user_weight
#     total_co2_emission = co2_emission_limestone + co2_emission_ggbs
#     energy_consumption_kwh = interpolated_result['Energy_Consumption(kWh)'] * user_weight
#     energy_cost = energy_consumption_kwh * energy_price_per_kwh
#     material_cost = (limestone_ratio * limestone_price_per_ton/1000 * user_weight) + (ggbs_ratio * ggbs_price_per_ton/1000 * user_weight)

#     # optimized_limestone_ratio,  optimized_ggbs_ratio, optimized_total_co2_emission, optimized_compressive_strength, optimized_energy_consumption_kwh, optimized_energy_cost
#     optimized_ggbs_ratio = bayesian_optimization(optimized_select_1, optimized_select_2)
#     optimized_limestone_ratio = 1 - optimized_ggbs_ratio
#     optimized_interpolated_result = interpolate_data(optimized_limestone_ratio)
#     optimized_total_co2_emission = optimized_interpolated_result['Total_CO2_Emission(kg)'] * user_weight
#     new_input = pd.DataFrame({
#     'Cement': [550 * (1 - optimized_ggbs_ratio)],
#     'Fly Ash': [80],
#     'Water': [220],
#     'Age': [28],
#     'Total Binder': [550],
#     'Water-Binder Ratio': [0.4],
#     'Binder Ratio': [ggbs_ratio],
#     })
#     optimized_compressive_strength = model.predict(new_input)[0]
#     optimized_energy_consumption_kwh = optimized_interpolated_result['Energy_Consumption(kWh)'] * user_weight
#     optimized_energy_cost = optimized_energy_consumption_kwh * energy_price_per_kwh
    
#     # 그래프
#     plot_optimization_graph(optimized_select_1, optimized_select_2)

#     ##### 값을 출력 #####
#     print(f"CO2 Emission (Limestone): {co2_emission_limestone:.2f} kg")
#     print(f"CO2 Emission (GGBS): {co2_emission_ggbs:.2f} kg")
#     print(f"Total CO2 Emission: {total_co2_emission:.2f} kg")
#     print(f"Energy Consumption: {energy_consumption_kwh:.2f} kWh")
#     print(f"Energy Cost: {energy_cost:.2f} USD")
#     print(f"Material Cost: {material_cost:.2f} USD")
#     print(f"Baseline CO2 Emission: {baseline_co2_emission:.2f}%")
#     print(f"Baseline Energy Consumption: {baseline_energy_consumption:.2f}%")
#     print(f"Baseline Energy Cost: {baseline_energy_cost:.2f}%")
#     print(f"Compressive Strength: {compressive_strength:.2f} MPa")
#     print(f"Baseline Compressive Strength: {baseline_compressive_strength:.2f}%")
    
#     print(f"Optimized Limestone Ratio: {optimized_limestone_ratio:.2f}")
#     print(f"Optimized GGBS Ratio: {optimized_ggbs_ratio:.2f}")
#     print(f"Optimized Total CO2 Emission: {optimized_total_co2_emission:.2f} kg")
#     print(f"Optimized Compressive Strength: {optimized_compressive_strength:.2f} MPa")
#     print(f"Optimized Energy Consumption: {optimized_energy_consumption_kwh:.2f} kWh")
#     print(f"Optimized Energy Cost: {optimized_energy_cost:.2f} USD")


###### Unreal Engin -> Simpy ######

# 넘어오는 데이터 예시
# simulation_data = """
# {
#   "status": "success",
#   "message": "Simulation updated successfully.",
#   "updated_simulation_data": {
#     "simulation_id": "20241205_001",
#     "user_input": {
#       "limestone_ratio": 0.4,
#       "ggbs_ratio": 0.6
#     },
#     "environmental_factors": {
#       "temperature": 25.0,
#       "pressure": 101.3
#     },
#     "user_weight": 50,
#     "optimized_Select_1": 0,
#     "optimized_Select_2": 2,
#     "calculation_results": {
#       "co2_emission_limestone": 15008.58,
#       "co2_emission_ggbs": 3411.02,
#       "total_co2_emission": 18419.60,
#       "energy_consumption_kwh": 25.58,
#       "total_cost": 2558
#     }
#   }
# }
# """

# # JSON 데이터를 파싱
# data = json.loads(simulation_data)
# ##### 시뮬레이션 실행 #####

# env = simpy.Environment()
# env.process(simulate(env, 
#     ##### request 변수(필수) #####
#     request_status=data["status"],
#     simulation_id=data["updated_simulation_data"]["simulation_id"],
#     limestone_ratio=data["updated_simulation_data"]["user_input"]["limestone_ratio"],
#     ggbs_ratio=data["updated_simulation_data"]["user_input"]["ggbs_ratio"],
#     user_weight=data["updated_simulation_data"]["user_weight"],
#     optimized_select_1=data["updated_simulation_data"]["optimized_Select_1"],
#     optimized_select_2=data["updated_simulation_data"]["optimized_Select_2"],
#     before_co2_emission_limestone=data['updated_simulation_data']['calculation_results']['co2_emission_limestone'],
#     before_co2_emission_ggbs=data['updated_simulation_data']['calculation_results']['co2_emission_ggbs'],
#     before_total_co2_emission=data['updated_simulation_data']['calculation_results']['total_co2_emission'],
#     before_energy_consumption_kwh=data['updated_simulation_data']['calculation_results']['energy_consumption_kwh'],
#     before_total_cost=data['updated_simulation_data']['calculation_results']['total_cost'],
#     ##### request 변수(선택) #####
#     #temperature = data["updated_simulation_data"]["environmental_factors"]["temperature"]
#     #pressure = data["updated_simulation_data"]["environmental_factors"]["pressure"]
# ))
# env.run()
