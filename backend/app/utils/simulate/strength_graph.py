import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os


# CSV 파일에서 데이터 불러오기
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '1ton_total_simulation.csv')
data_df = pd.read_csv(csv_path)

# 보간을 위한 함수 정의
def interpolate_data(limestone_ratio):
    limestone_keys = data_df['Limestone_Ratio']
    interpolated_values = {}
    for column in data_df.columns[2:]:
        y = data_df[column]
        interpolated_values[column] = np.interp(limestone_ratio, limestone_keys, y)
    return interpolated_values

# 저장된 모델 불러오기
model = joblib.load('/content/drive/MyDrive/model1206.joblib')

# GGBS 비율을 0에서 1까지 변경하며 결과 계산
ggbs_ratios = np.linspace(0, 1, 100)
results = {
    "GGBS Ratio": [],
    "Total CO2 Emission": [],
    "Energy Consumption": [],
    "Compressive Strength": []
}

for ggbs_ratio in ggbs_ratios:
    limestone_ratio = 1-ggbs_ratio
    interpolated_result = interpolate_data(limestone_ratio)
    total_co2_emission = interpolated_result['Total_CO2_Emission(kg)'] * 500
    energy_consumption = interpolated_result['Energy_Consumption(kWh)'] * 500
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
    print(ggbs_ratio, compressive_strength," ")


    results["GGBS Ratio"].append(ggbs_ratio)
    results["Total CO2 Emission"].append(total_co2_emission)
    results["Energy Consumption"].append(energy_consumption)
    results["Compressive Strength"].append(compressive_strength)

# 데이터프레임으로 변환
results_df = pd.DataFrame(results)

# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(10, 6))

# 첫 번째 y축: Total CO2 Emission (로그 스케일 적용)
ax1.plot(results_df["GGBS Ratio"], results_df["Total CO2 Emission"], color="blue", label="Total CO2 Emission (kg)")
ax1.set_xlabel("GGBS Ratio")
ax1.set_ylabel("Total CO2 Emission (kg)", color="blue")
ax1.set_yscale('log')  # 로그 스케일 설정
ax1.tick_params(axis="y", labelcolor="blue")

# 두 번째 y축: Energy Consumption
ax2 = ax1.twinx()
ax2.plot(results_df["GGBS Ratio"], results_df["Energy Consumption"], color="orange", label="Energy Consumption (kWh)")
ax2.set_ylabel("Energy Consumption (kWh)", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")

ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))
ax3.plot(results_df["GGBS Ratio"], results_df["Compressive Strength"], color="green", label="Compressive Strength (MPa)")
ax3.set_ylabel("Compressive Strength (MPa)", color="green")
ax3.tick_params(axis="y", labelcolor="green")
ax3.set_ylim(20, 70)  # y축 범위를 39~41로 고정

# 제목 설정
fig.suptitle("Effect of GGBS Ratio on CO2 Emission, Energy Consumption, and Compressive Strength")
fig.tight_layout()
plt.show()