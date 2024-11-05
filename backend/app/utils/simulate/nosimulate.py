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

# 결과 계산 및 출력 함수
def calculate_and_print_results(limestone_ratio, ggbs_ratio, weight_ton):
    # 보간된 데이터 가져오기
    interpolated_result = interpolate_data(limestone_ratio)
    
    # 무게에 따른 결과 조정
    co2_emission_limestone = interpolated_result['CO2_Emission_Limestone(kg)'] * weight_ton
    co2_emission_ggbs = interpolated_result['CO2_Emission_GGBS(kg)'] * weight_ton
    total_co2_emission = interpolated_result['Total_CO2_Emission(kg)'] * weight_ton
    energy_consumption = interpolated_result['Energy_Consumption(kWh)'] * weight_ton
    total_cost = interpolated_result['Total_Cost(KRW)'] * weight_ton

    # 결과 출력
    print(f"\n--- Simulation Result ---")
    print(f"Limestone Ratio: {limestone_ratio}")
    print(f"GGBS Ratio: {ggbs_ratio}")
    print(f"Weight: {weight_ton} ton")
    print(f"CO2 Emission (Limestone): {co2_emission_limestone:.2f} kg")
    print(f"CO2 Emission (GGBS): {co2_emission_ggbs:.2f} kg")
    print(f"Total CO2 Emission: {total_co2_emission:.2f} kg")
    print(f"Energy Consumption: {energy_consumption:.2f} kWh")
    print(f"Total Cost (KRW): {total_cost:.2f}")

# 시뮬레이션 반복을 위한 while 루프
while True:
    # 사용자가 입력한 값을 통해 시뮬레이션 시작
    try:
        user_limestone_ratio = float(input("Limestone 비율을 입력하세요 (0.0 ~ 1.0): "))
        if user_limestone_ratio < 0 or user_limestone_ratio > 1:
            print("비율은 0.0에서 1.0 사이여야 합니다.")
            continue
    except ValueError:
        print("올바른 숫자를 입력해 주세요.")
        continue

    user_ggbs_ratio = 1 - user_limestone_ratio  # Limestone과 GGBS의 비율의 합이 1이므로 GGBS는 자동 계산
    
    try:
        user_weight = float(input("시멘트의 총 무게를 입력하세요 (톤 단위): "))
        if user_weight <= 0:
            print("무게는 0보다 커야 합니다.")
            continue
    except ValueError:
        print("올바른 숫자를 입력해 주세요.")
        continue

    # 결과 계산 및 출력
    calculate_and_print_results(user_limestone_ratio, user_ggbs_ratio, user_weight)

    # 시뮬레이션 반복 여부 확인
    repeat = input("\n다시 시뮬레이션을 하시겠습니까? (y/n): ").strip().lower()
    if repeat != 'y':
        break
