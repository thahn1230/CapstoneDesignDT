import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
import seaborn as sns
import joblib
import os


# 데이터 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'ConcreteData.csv')
data = pd.read_csv(csv_path)

# 컬럼 이름 간소화
data = data.rename(columns={
    'Cement (component 1)(kg in a m^3 mixture)': 'Cement',
    'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'Blast Furnace Slag',
    'Fly Ash (component 3)(kg in a m^3 mixture)': 'Fly Ash',
    'Water  (component 4)(kg in a m^3 mixture)': 'Water',
    'Concrete compressive strength(MPa, megapascals) ': 'Compressive Strength',
    'Age (day)': 'Age',
    'Superplasticizer (component 5)(kg in a m^3 mixture)': 'Superplasticizer',
    'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'Coarse Aggregate',
    'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'Fine Aggregate',
})

# 추가적인 변수 생성
data['Total Binder'] = data['Cement'] + data['Blast Furnace Slag'] + data['Fly Ash']
data['Water-Binder Ratio'] = data['Water'] / data['Total Binder']
data['Binder Ratio'] = data['Blast Furnace Slag'] / (data['Cement'] + data['Blast Furnace Slag'])

# X, y 정의
X = data.drop(columns=['Compressive Strength'])
y = data['Compressive Strength']

# 이상치 제거 함수 정의 (IQR 방식)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)  # 1사분위수
    Q3 = df[column].quantile(0.75)  # 3사분위수
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# X와 y 결합
#data_combined = pd.concat([X, y], axis=1)

# X의 각 열에 대해 이상치 제거 (X에 대해서만)
# for col in X.columns:
#     data_combined = remove_outliers(data_combined, col)

# 이상치 제거 후 X와 y 분리
# X = data.drop(columns=['Compressive Strength'])
# y = data['Compressive Strength']

# 데이터 정규화 (X만 정규화)
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Normalized 데이터셋 기본 통계 출력
print("X Normalized Basic Statistics:")
print(X_normalized.describe())

# Box Plot 생성 (X_normalized 데이터 사용)
plt.figure(figsize=(12, 8))
boxplot = plt.boxplot(X_normalized.values, patch_artist=True)
colors = ['#FFDDC1', '#FFABAB', '#FFC3A0', '#D5AAFF', '#85E3FF',
          '#B9FBC0', '#FF9CEE', '#A0E7E5', '#B4F8C8', '#FBE7C6', '#C1FFD7', '#FFFFC1']
for patch, color in zip(boxplot['boxes'], colors * (len(X_normalized.columns) // len(colors) + 1)):
    patch.set_facecolor(color)
plt.title("Variable Distribution (Normalized X Box Plot)")
plt.xlabel("Variables")
plt.ylabel("Range (Normalized)")
plt.xticks(ticks=range(1, len(X_normalized.columns) + 1), labels=X_normalized.columns, rotation=45)
plt.show()

# 상관관계 히트맵 (X_normalized 데이터 사용)
plt.figure(figsize=(10, 8))
correlation_matrix_normalized = X_normalized.corr()
sns.heatmap(correlation_matrix_normalized, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title("Correlation Heatmap (Normalized X)")
plt.show()

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Gradient Boosting 모델 정의
model = GradientBoostingRegressor(
    learning_rate=0.42,
    loss='huber',
    max_depth=3,
    min_samples_leaf=3,
    min_samples_split=5,
    n_estimators=967,
    random_state=42
)

# 모델 학습
model.fit(X_train, y_train)

# 전체 Feature Importance 시각화
plt.figure(figsize=(8, 6))
feature_importances = model.feature_importances_
plt.barh(X.columns, feature_importances, color='skyblue')
plt.title("Feature Importances (All Features)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.grid(True)
plt.show()

# Feature Inputs 수에 따른 정확도 분석
r2_scores = []
input_counts = range(1, X.shape[1] + 1)

for num_features in input_counts:
    rfe = RFE(estimator=model, n_features_to_select=num_features)
    rfe.fit(X_train, y_train)

    # 선택된 피처로 데이터 구성
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    # 모델 학습 및 예측
    model.fit(X_train_rfe, y_train)
    y_pred = model.predict(X_test_rfe)

    # R^2 스코어 저장
    r2_scores.append(r2_score(y_test, y_pred))

# Feature Inputs 수에 따른 정확도 시각화
plt.figure(figsize=(8, 6))
plt.plot(input_counts, r2_scores, marker='o', linestyle='-', color='b', label='R² Score')
plt.title("R² Score vs Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("R² Score")
plt.grid(True)
plt.legend()
plt.show()

# 최적 Feature Inputs 수로 결과 출력
optimal_features = input_counts[np.argmax(r2_scores)]
print(f"Optimal Number of Features: {optimal_features}")

# 최적 Feature Inputs 수로 RFE 적용
rfe = RFE(estimator=model, n_features_to_select=optimal_features)
rfe.fit(X_train, y_train)
X_train_optimal = rfe.transform(X_train)
X_test_optimal = rfe.transform(X_test)

# 최적 Feature Inputs 수로 모델 학습 및 예측
model.fit(X_train_optimal, y_train)
#joblib.dump(model, '/content/drive/My Drive/model1206.joblib')
y_pred_optimal = model.predict(X_test_optimal)

# 최적 Feature Inputs 수 결과 출력
mse = mean_squared_error(y_test, y_pred_optimal)
mae = mean_absolute_error(y_test, y_pred_optimal)
r2 = r2_score(y_test, y_pred_optimal)

print(f"Optimal Model - Mean Squared Error: {mse:.2f}")
print(f"Optimal Model - Mean Absolute Error: {mae:.2f}")
print(f"Optimal Model - R²: {r2:.2f}")
