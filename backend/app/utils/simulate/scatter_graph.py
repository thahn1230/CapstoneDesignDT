import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# 데이터 불러오기
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'ConcreteData.csv')
data = pd.read_csv(csv_path)

# 컬럼명 정리
data = data.rename(columns={
    'Cement (component 1)(kg in a m^3 mixture)': 'Cement',
    'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'Blast Furnace Slag',
    'Fly Ash (component 3)(kg in a m^3 mixture)': 'Fly Ash',
    'Water  (component 4)(kg in a m^3 mixture)': 'Water',
    'Concrete compressive strength(MPa, megapascals) ': 'Compressive Strength',
    'Age (day)': 'Age',
    'Superplasticizer (component 5)(kg in a m^3 mixture)': 'super',
    'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'coarse',
    'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'fine',
})

# 필요한 특성 추가 계산
data['Total Binder'] = data['Cement'] + data['Blast Furnace Slag']
data['Water-Binder Ratio'] = data['Water'] / data['Total Binder']
data['Binder Ratio'] = data['Blast Furnace Slag'] / (data['Cement'] + data['Blast Furnace Slag'])
data = data[data['Age'] == 28]

# GGBS Ratio (Binder Ratio)를 0.1 단위로 그룹화
data['GGBS Ratio Group'] = pd.cut(data['Binder Ratio'], bins=np.arange(0, 0.8, 0.1), labels=np.round(np.arange(0.1, 0.8, 0.1), 1), include_lowest=True)

# 그룹별 평균값 계산
grouped = data.groupby(['GGBS Ratio Group', 'Water-Binder Ratio']).mean().reset_index()

# 그래프 그리기
plt.figure(figsize=(10, 6))

# GGBS Ratio 그룹별로 점만 표시
for ggbs_ratio, subset in grouped.groupby('GGBS Ratio Group'):
    #plt.plot(subset['Water-Binder Ratio'], subset['Compressive Strength'], label=f'GGBS: {ggbs_ratio}')
    plt.scatter(subset['Water-Binder Ratio'], subset['Compressive Strength'], label=f'GGBS: {ggbs_ratio}', alpha=0.7)
plt.xlim(0, 1)
# 그래프 설정
plt.xlabel("Water-Binder Ratio (W/B)")
plt.ylabel("Compressive Strength (MPa)")
plt.title("Compressive Strength vs Water-Binder Ratio for Different GGBS Ratios")
plt.legend(title="GGBS Ratio")
plt.grid(True)
plt.show()
