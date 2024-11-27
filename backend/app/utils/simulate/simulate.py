import simpy
import pandas as pd
import numpy as np

# CSV 파일에서 데이터 불러오기
csv_path = '1ton_total_simulation.csv'
data_df = pd.read_csv(csv_path)

# 보간을 위한 함수 정의
def interpolate_data(limestone_ratio):
    # Limestone 비율에 따른 보간을 수행
    limestone_keys = data_df['Limestone_Ratio']
    # 각 열에 대해 보간 수행
    interpolated_values = {}
    for column in data_df.columns[2:]:
        y = data_df[column]
        interpolated_values[column] = np.interp(limestone_ratio, limestone_keys, y)
    
    return interpolated_values

# SimPy 시뮬레이션 함수 (generator로 변경)
# weight_ton : csv 1톤 기준, 시멘트 몇톤? DT Layer....
def simulate(env, limestone_ratio, ggbs_ratio, weight_ton):
    # Limestone과 GGBS의 비율의 합이 1이 아니면 종료
    # if limestone_ratio < 0 or limestone_ratio > 1 :
    #     print("비율을 확인하세요")
    #     return
    
    # 시뮬레이션을 위한 최소 시간 대기 (SimPy 요구사항)
    yield env.timeout(0)

    # 보간된 데이터 가져오기
    interpolated_result = interpolate_data(limestone_ratio)
    
    # 무게에 따른 결과 조정
    co2_emission_limestone = interpolated_result['CO2_Emission_Limestone(kg)'] * weight_ton
    co2_emission_ggbs = interpolated_result['CO2_Emission_GGBS(kg)'] * weight_ton
    total_co2_emission = interpolated_result['Total_CO2_Emission(kg)'] * weight_ton
    energy_consumption = interpolated_result['Energy_Consumption(kWh)'] * weight_ton
    total_cost = interpolated_result['Total_Cost(KRW)'] * weight_ton

    # 결과 출력
    # print(f"\n--- Simulation Result ---")
    # print(f"Limestone Ratio: {limestone_ratio}")
    # print(f"GGBS Ratio: {ggbs_ratio}")
    # print(f"Weight: {weight_ton} ton")
    # print(f"CO2 Emission (Limestone): {co2_emission_limestone:.2f} kg")
    # print(f"CO2 Emission (GGBS): {co2_emission_ggbs:.2f} kg")
    # print(f"Total CO2 Emission: {total_co2_emission:.2f} kg")
    # print(f"Energy Consumption: {energy_consumption:.2f} kWh")
    # print(f"Total Cost (KRW): {total_cost:.2f}")

    total_data = {
        "limestone_ratio": {limestone_ratio},
        "ggbs_ratio": {ggbs_ratio},
        "co2_emission_limestone": {co2_emission_limestone},
        "co2_emission_ggbs": {co2_emission_ggbs},
        "energy_consumption_kwh": {energy_consumption},
        "total_cost": {total_cost}
    }

    return total_data

# SimPy 환경 설정
env = simpy.Environment()

while True:
    # 사용자가 입력한 값을 통해 시뮬레이션 시작
    user_limestone_ratio = float(input("Limestone 비율을 입력하세요 (0.0 ~ 1.0): "))
    user_ggbs_ratio = 1 - user_limestone_ratio  # Limestone과 GGBS의 비율의 합이 1이므로 GGBS는 자동 계산
    user_weight = float(input("시멘트의 총 무게를 입력하세요 (톤 단위): "))

    # 시뮬레이션 실행
    env.process(simulate(env, user_limestone_ratio, user_ggbs_ratio, user_weight))
    env.run()

    # 시뮬레이션 반복 여부 확인
    repeat = input("\n다시 시뮬레이션을 하시겠습니까? (y/n): ").strip().lower()
    if repeat != 'y':
        break
