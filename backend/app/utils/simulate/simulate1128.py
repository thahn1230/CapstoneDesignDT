import simpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
import joblib
import os

# 가격정의
limestone_price_per_ton = 8168
ggbs_price_per_ton = 22160
energy_price_per_kwh = 120

# model = joblib.load('/content/drive/MyDrive/strength_predict_Model.joblib')
# data_df = pd.read_csv('/content/drive/MyDrive/1ton_total_simulation.csv')


# 현재 파일의 디렉토리를 기준으로 경로 설정
base_dir = os.path.dirname(__file__)

# 모델과 데이터 파일 경로 설정
model_path = os.path.join(base_dir, "strength_predict_Model.joblib")
csv_path = os.path.join(base_dir, "1ton_total_simulation.csv")

# 모델 로드
model = joblib.load(model_path)

# 데이터 로드
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

def simulate(env, limestone_ratio, ggbs_ratio):
    if limestone_ratio < 0 or limestone_ratio > 1:
        print("비율을 확인하세요")
        return

    yield env.timeout(0)  # SimPy에서 시간 흐름 처리

    # 데이터 보간 및 계산
    interpolated_result = interpolate_data(limestone_ratio)
    material_cost = (limestone_ratio * limestone_price_per_ton) + (ggbs_ratio * ggbs_price_per_ton)
    total_co2_emission = interpolated_result['Total_CO2_Emission(kg)'] * 1000
    energy_consumption = interpolated_result['Energy_Consumption(kWh)'] * 1000
    energy_cost = energy_consumption * energy_price_per_kwh * 1000

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
    reduction_percentage = ((total_co2_emission - optimal_co2_emission) / total_co2_emission) * 100


    # 결과 출력
    print(f"\n--- Simulation Result ---")
    print(f"Limestone Ratio: {limestone_ratio:.2f}")
    print(f"GGBS Ratio: {ggbs_ratio:.2f}")
    print(f"Total CO2 Emission: {total_co2_emission:.2f} kg")
    print(f"reduction_percentage : {reduction_percentage:.2f}%")
    print(f"Energy Consumption: {energy_consumption:.2f} kWh")
    print(f"Predicted Compressive Strength: {compressive_strength:.2f} MPa")
    print(f"Material Cost: {material_cost:.2f} KRW")
    print(f"Energy Cost: {energy_cost:.2f} KRW")

env = simpy.Environment()

# while True:
#     # 사용자 입력 받기1

#     user_ggbs_ratio = float(input("ggbs 비율을 입력하세요 (0.0 ~ 1.0): "))
#     user_limestone_ratio = 1 - user_ggbs_ratio

#     # 시뮬레이션 실행
#     env.process(simulate(env, user_limestone_ratio, user_ggbs_ratio))
#     env.run()

#     # 시뮬레이션 반복 여부 확인
#     repeat = input("\n다시 시뮬레이션을 하시겠습니까? (y/n): ").strip().lower()
#     if repeat != 'y':
#         break
