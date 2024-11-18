import simpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
import os

# 변수 정의 (필요한 값을 미리 설정해야 함)
limestone_price_per_ton = 500  # Limestone 가격 (예: KRW/ton)
ggbs_price_per_ton = 1000      # GGBS 가격 (예: KRW/ton)
energy_price_per_kwh = 150     # 에너지 비용 (예: KRW/kWh)

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "model_new2.joblib")
model = joblib.load(model_path)
# 보간 함수 (샘플 데이터로 작성, 실제 데이터는 제공되어야 함)
def interpolate_data(limestone_ratio):
    # 보간된 데이터 예시 (실제 데이터로 대체 필요)
    return {
        'Total_CO2_Emission(kg)': 800 - 300 * limestone_ratio,
        'Energy_Consumption(kWh)': 200 - 100 * limestone_ratio
    }

# 시뮬레이션 함수
def simulate(env, limestone_ratio, ggbs_ratio, weight_ton):
    if limestone_ratio < 0 or limestone_ratio > 1:
        print("비율을 확인하세요")
        return

    yield env.timeout(0)  # SimPy에서 시간 흐름 처리

    # 데이터 보간 및 계산
    interpolated_result = interpolate_data(limestone_ratio)
    material_cost = (limestone_ratio * limestone_price_per_ton) + (ggbs_ratio * ggbs_price_per_ton)
    total_co2_emission = interpolated_result['Total_CO2_Emission(kg)'] * weight_ton
    energy_consumption = interpolated_result['Energy_Consumption(kWh)'] * weight_ton
    energy_cost = energy_consumption * energy_price_per_kwh

    # 압축 강도 계산 (모델을 사용)
    new_input = pd.DataFrame({'Binder Ratio': [limestone_ratio], 'Total Binder': [300], 'Water-Binder Ratio': [0.4]})
    #new_input_scaled = scaler.transform(new_input)

    # 모델로 예측
    compressive_strength = model.predict(new_input)[0]
    

    # 결과 출력
    print(f"\n--- Simulation Result ---")
    print(f"Limestone Ratio: {limestone_ratio:.2f}")
    print(f"GGBS Ratio: {ggbs_ratio:.2f}")
    print(f"Weight: {weight_ton} ton")
    print(f"Total CO2 Emission: {total_co2_emission:.2f} kg")
    print(f"Energy Consumption: {energy_consumption:.2f} kWh")
    print(f"Predicted Compressive Strength: {compressive_strength:.2f} MPa")
    print(f"Material Cost: {material_cost:.2f} KRW")
    print(f"Energy Cost: {energy_cost:.2f} KRW")

# Bayesian Optimization의 Objective Function
def objective(params):
    ggbs_ratio, limestone_ratio = params
    if ggbs_ratio + limestone_ratio != 1:
        limestone_ratio = 1 - ggbs_ratio

    # Simulate 결과 생성
    interpolated_result = interpolate_data(limestone_ratio)
    material_cost = (limestone_ratio * limestone_price_per_ton) + (ggbs_ratio * ggbs_price_per_ton)
    total_co2_emission = interpolated_result['Total_CO2_Emission(kg)']
    energy_consumption = interpolated_result['Energy_Consumption(kWh)']
    energy_cost = energy_consumption * energy_price_per_kwh

    # 압축 강도 계산
    new_input = pd.DataFrame({'Binder Ratio': [limestone_ratio], 'Total Binder': [300], 'Water-Binder Ratio': [0.4]})
    # new_input_scaled = scaler.transform(new_input)
    compressive_strength = model.predict(new_input)[0]

    # 가중치 적용
    weight_strength = 15.0
    weight_co2 = 1.0
    weight_energy = 0.01
    weight_material_cost = 0.01
    weight_energy_cost = 0.01

    target = (
        -weight_strength * compressive_strength +  # 압축 강도 최대화
        weight_co2 * total_co2_emission +          # CO2 배출 최소화
        weight_energy * energy_consumption +       # 에너지 소비 최소화
        weight_material_cost * material_cost +     # 재료 비용 최소화
        weight_energy_cost * energy_cost           # 에너지 비용 최소화
    )

    return target

# 최적화 및 결과 시각화 함수
def plot_optimization_results():
    search_space = [
        Real(0, 1, name='ggbs_ratio'),
        Real(0, 1, name='limestone_ratio')
    ]

    opt_result = gp_minimize(objective, search_space, n_calls=50, random_state=42)
    optimal_ggbs_ratio, optimal_limestone_ratio = opt_result.x
    optimal_limestone_ratio = 1 - optimal_ggbs_ratio  # 비율 합이 1로 유지

    print("\n--- Optimal Ratios ---")
    print(f"GGBS Ratio: {optimal_ggbs_ratio:.2f}")
    print(f"Limestone Ratio: {optimal_limestone_ratio:.2f}")

    # 최적화된 비율로 시뮬레이션 실행
    env.process(simulate(env, optimal_limestone_ratio, optimal_ggbs_ratio, 1))  # 무게 1톤 기본값
    env.run()

    # 그래프 시각화
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 1, 100)
    plt.plot(x, [objective([xi, 1 - xi]) for xi in x], label='Objective Value', color='blue')
    plt.scatter([optimal_ggbs_ratio], [objective([optimal_ggbs_ratio, optimal_limestone_ratio])], color='red', label='Optimal Point')
    plt.xlabel("GGBS Ratio")
    plt.ylabel("Objective Value")
    plt.legend()
    plt.show()

# SimPy 환경 설정
env = simpy.Environment()

while True:
    # 사용자 입력 받기1

    user_ggbs_ratio = float(input("ggbs 비율을 입력하세요 (0.0 ~ 1.0): "))
    user_limestone_ratio = 1 - user_ggbs_ratio
    user_weight = float(input("시멘트의 총 무게를 입력하세요 (톤 단위): "))

    # 시뮬레이션 실행
    env.process(simulate(env, user_limestone_ratio, user_ggbs_ratio, user_weight))
    env.run()

    # Bayesian Optimization 실행 여부
    optimize = input("\nBayesian Optimization으로 최적 비율을 찾으시겠습니까? (y/n): ").strip().lower()
    if optimize == 'y':
        plot_optimization_results()

    # 시뮬레이션 반복 여부 확인
    repeat = input("\n다시 시뮬레이션을 하시겠습니까? (y/n): ").strip().lower()
    if repeat != 'y':
        break
