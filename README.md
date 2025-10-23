# 제주 특산물 가격 예측 프로젝트 (Dacon Jeju Price Prediction)
https://dacon.io/competitions/official/236176/overview/description

---

## 1. 데이터 로드 & 기본 확인

```python
import pandas as pd
import numpy as np

# 1. 데이터 로드
df = pd.read_csv("./train.csv")

# 1-1. 기본 정보 확인
print("[기본 정보 확인]")
print("-" * 60)
print(f"데이터 크기: {df.shape}")
print("\n데이터 타입 및 결측치 요약:")
print(df.info())
print("\n상위 5개 행:")
print(df.head())

# 1-2. 결측치 제거
df_clean = df.dropna().reset_index(drop=True)
print("\n[결측치 제거 결과]")
print(f"제거 전 데이터 크기: {df.shape}")
print(f"제거 후 데이터 크기: {df_clean.shape}")

# 1-4. 수치형 컬럼 기본 통계
print("\n[기초 통계 요약 (수치형 변수)]")
print(df_clean.describe().T)

# 1-5. 문자열형(범주형) 컬럼 고유값 개수
print("\n[범주형 변수 고유값 요약]")
cat_cols = df_clean.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"{col}: {df_clean[col].nunique()}개 (예시: {df_clean[col].unique()[:5]})")
```

---

## 2. 탐색적 데이터 분석 (EDA)

### (1) 상관관계 Heatmap

```python
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

![heatmap](https://github.com/user-attachments/assets/55c55f34-46fe-4485-949f-0db83bad6095)

**해석:** 공급량과 가격 간에는 음(-)의 상관관계가 존재하며, 공급이 늘어날수록 단가가 하락하는 수급 구조를 보인다.

---

### (2) Timestamp별 가격 추이

```python
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x='timestamp', y='price(원/kg)')
plt.title('Price Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Price (원/kg)')
plt.show()
```

![timeseries](https://github.com/user-attachments/assets/13c185d4-0266-40f5-97eb-ba8c85da627b)

**해석:**  
가격은 명확한 주기성을 가진 파형 형태로 나타나며, 일정 주기마다 상승·하락 패턴이 반복된다.  
이는 계절적 수요·공급 변화 또는 출하 시기와 밀접한 연관이 있다.  
진폭이 일정하지 않아 변동성이 크며, 공급 부족 구간에서는 가격 급등이 나타난다.

---

### (3) 지역별 평균 가격

```python
plt.figure(figsize=(12,6))
sns.barplot(data=df, x='location', y='price(원/kg)', hue='item', estimator=np.mean, errorbar=None)
plt.title('Average Price of Each Item by Location')
plt.xlabel('Location')
plt.ylabel('Average Price (원/kg)')
plt.legend(title='Item', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

![bar](https://github.com/user-attachments/assets/281ecfa8-493e-4d44-bbf3-5aa7e6b34556)

**해석:** TG 품목이 다른 품목 대비 압도적으로 높은 단가를 유지하고 있으며, 일부 지역에 프리미엄 품목 집중도가 존재한다.

---

## 3. 군집화 (KMeans 파생변수 추가)

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

num_df = df.select_dtypes(include=np.number).dropna(axis=1)
scaler = StandardScaler()
scaled = scaler.fit_transform(num_df)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled)

print(df.groupby('cluster')[['price(원/kg)', 'supply(kg)']].mean())
```

**결과 요약:**  
- cluster 0: 낮은 가격·높은 공급 (공급 과잉)  
- cluster 1: 중간 수준 균형형  
- cluster 2: 높은 가격·낮은 공급 (프리미엄)  

---

## 4. 회귀 분석 (OLS & Ridge)

### (1) OLS 회귀분석

![ols](https://github.com/user-attachments/assets/d5fecb0e-2fd9-4e2b-a716-bb8445adf634)

**해석:**  
- R² = 0.795 → 약 80% 설명력 확보  
- `supply(kg)`은 음(-)의 계수, `item_TG`는 양(+)의 계수로 가격 형성에 큰 영향  
- 일부 법인(B, F)과 지역(S)은 음(-) 방향 영향, TG 품목은 가격을 끌어올리는 요인  

---

### (2) Ridge 회귀분석

```python
ridge = Ridge(alpha=10, random_state=42)
ridge.fit(X, y)
y_pred = ridge.predict(X)
print(f"R²: {r2_score(y, y_pred):.4f}")
```

**해석:**  
Ridge는 다중공선성 완화 효과로 안정적인 예측력을 보여줬으며,  
과적합을 줄이면서 변수 간 의존성을 억제해 일반화 성능을 강화했다.

---

## 5. 지역별 ANOVA (집단 간 유의미한 차이 검증)

![anova](https://github.com/user-attachments/assets/b33f932a-6e43-47d5-8ad0-1675d6d4fe41)

**결과 해석:**  
- Shapiro-Wilk → 비정규 분포 (p<0.05)  
- Levene → 등분산성 깨짐 (p<0.05)  
- Welch ANOVA 결과: F≈96.9, p≈7.6e-23  
→ 지역 간 평균 가격 차이는 **통계적으로 유의**하지만, 경제적 규모는 작음 (np²=0.0017).

---

## 6. Boosting & Stacking (머신러닝 예측 모델)

![stack](https://github.com/user-attachments/assets/a58f3715-f8a5-4fa6-9a4b-77a5c3170875)

**구성:**  
- Base Layer: LightGBM, XGBoost, GradientBoosting  
- Meta Model: CatBoost  
- 성능: R² ≈ 0.80, RMSE 안정적

**Feature Importance 결과:**  
![importance](https://github.com/user-attachments/assets/c1593885-37c3-4f57-ab47-75a4c9796d51)

- 주요 영향 변수: `item`, `corporation`, `location`  
- `supply(kg)`보다 범주형 요인(거래 주체, 품목)이 결정적  

---

## 7. 종합 결론

이번 분석을 통해 제주 지역의 품목별 가격은 **시간적 요인보다 구조적 요인**에 의해 더 크게 영향을 받는다는 것이 확인되었다.  
공급량이 많을수록 가격은 하락하고, TG 품목은 프리미엄 상품으로서 지속적인 고가를 형성한다.  
지역 간 가격 격차는 통계적으로 유의하나, 실제 경제적 영향력은 미미했다.  
머신러닝 모델은 R² ≈ 0.8의 예측력을 보여주며, 가격은 **누가(법인), 어떤 품목을, 어느 지역에서 거래하느냐**에 의해 결정되는 구조임이 드러났다.
```
eju_Price_Prediction
 ┣ train.csv
 ┣ jeju_analysis.ipynb
 ┣ README.md
 ┗ requirements.txt
```

---

## Author

**윤해정 (Yoon Haejeong)**  
heajeongy@naver.com  
Data Analytics & Visualization / Python, SQL, Power BI  
[GitHub](https://github.com/heajeongy-design)
