# 제주 특산물 가격 예측 AI 경진대회 (Dacon_Jeju_Price_Prediction)
https://dacon.io/competitions/official/236176/overview/description

## 프로젝트 개요
본 프로젝트는 제주도 지역 품목별 가격 데이터를 기반으로 한 시각적·통계적 탐색 분석(EDA)과 머신러닝 기반 예측 모델링입니다.  
가격 변동 요인 분석 → 통계적 검정 → Stacking 기반 예측 모델 구축 및 Feature Importance 해석까지의 전 과정을 수행했습니다.

> **목적:**
> - 품목별(`item`) 가격(`price(원/kg)`) 변동의 주요 요인 분석
> - 지역(`location`), 법인(`corporation`) 간의 구조적 차이 파악
> - 시계열·공급·시장 구조 요인 기반의 예측 모델 구축

---

## 데이터 요약
| 컬럼명 | 설명 |
|--------|------|
| `ID` | 거래 식별자 |
| `timestamp` | 거래 일자 |
| `item` | 품목명 (BC, CB, CR, RD, TG 등) |
| `corporation` | 거래 법인 코드 (A~F) |
| `location` | 지역 코드 (J, S) |
| `supply(kg)` | 공급량 |
| `price(원/kg)` | 단가 |

- 전체 행: 59,397개  
- 결측값: 0건  
- `timestamp`는 datetime64 형식으로 변환

---

## 탐색적 데이터 분석 (EDA)

### (1) 상관관계 Heatmap
```python
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

![heatmap](https://github.com/user-attachments/assets/55c55f34-46fe-4485-949f-0db83bad6095)

- 공급량과 가격 간 뚜렷한 음(-)의 상관관계 → 공급 증가 시 단가 하락

---

### (2) 시계열별 가격 추이
```python
sns.lineplot(data=df, x='timestamp', y='price(원/kg)')
plt.title('Price Trend Over Time')
```

![timeseries](https://github.com/user-attachments/assets/13c185d4-0266-40f5-97eb-ba8c85da627b)

- 일정 주기로 급등락하는 파형 패턴 → 계절성 혹은 수요·공급 주기 반영  
- 진폭이 일정하지 않아 시장 변동성이 높은 것으로 보임

---

### (3) 지역별 평균 가격
```python
sns.barplot(data=df, x='location', y='price(원/kg)', hue='item', estimator=np.mean, errorbar=None)
plt.title('Average Price by Location and Item')
```

![bar](https://github.com/user-attachments/assets/281ecfa8-493e-4d44-bbf3-5aa7e6b34556)

- TG 품목의 평균 가격이 압도적으로 높음 → 프리미엄 상품 가능성  
- CB·CR 등은 지역 간 차이가 미미함

---

## 군집화 (KMeans)
```python
num_df = df.select_dtypes(include=np.number).dropna(axis=1)
scaler = StandardScaler()
scaled = scaler.fit_transform(num_df)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled)
```

**결과:**
```python
print(df.groupby('cluster')[['price(원/kg)', 'supply(kg)']].mean())
```

| cluster | 평균 가격 | 평균 공급량 |
|----------|------------|-------------|
| 0 | 낮은 가격, 높은 공급 | 공급 과잉 그룹 |
| 1 | 중간 수준 | 균형 그룹 |
| 2 | 높은 가격, 낮은 공급 | 프리미엄 그룹 |

---

## 회귀 분석 (OLS & Ridge)

### (1) OLS 회귀분석
```python
X = pd.get_dummies(df[['supply(kg)', 'item', 'corporation', 'location', 'cluster']], drop_first=True)
y = df['price(원/kg)']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
```

![ols](https://github.com/user-attachments/assets/d5fecb0e-2fd9-4e2b-a716-bb8445adf634)

- R² = 0.795 → 80% 설명력 확보  
- `supply(kg)`은 음(-), `item_TG`는 양(+) → 공급 많을수록 가격 하락, TG 품목은 프리미엄 효과

### (2) Ridge 회귀
```python
ridge = Ridge(alpha=10, random_state=42)
ridge.fit(X, y)
y_pred = ridge.predict(X)
print(f"R²: {r2_score(y, y_pred):.4f}")
```

- 다중공선성을 완화하여 안정성 향상  
- 주요 양(+) 변수: `item_TG`, `cluster_2`  
- 주요 음(-) 변수: `corporation_B`, `location_S`

---

## 지역 간 평균 비교 (Welch ANOVA)
```python
anova_result = welch_anova(dv='price(원/kg)', between='location', data=df)
print(anova_result)
```

![anova](https://github.com/user-attachments/assets/b33f932a-6e43-47d5-8ad0-1675d6d4fe41)

- F(1, 59395)=96.9, p=7.58e-23 → 지역 간 평균 차이 유의함  
- 효과크기(np²)=0.0017 → 실질적 차이는 작음  
- 지역별 물류비/수요 구조 차이가 가격 형성에 미세한 영향을 줌

---

## 머신러닝 모델링 (Boosting + Stacking)

### 구성
- 전방(Base): LightGBM, XGBoost, GradientBoosting
- 후방(Meta): CatBoost (Meta Model)

```python
stack_model = StackingRegressor(
    estimators=[('lgb', LGBMRegressor()), ('xgb', XGBRegressor()), ('gbm', GradientBoostingRegressor())],
    final_estimator=CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, loss_function='RMSE', verbose=False),
    passthrough=True
)
stack_model.fit(X_train_scaled, y_train)
```

**성능:**
```python
R² = 0.80, RMSE ≈ 안정적
```

---

## Feature Importance (CatBoost)

```python
meta_model = stack_model.final_estimator_
importances = meta_model.get_feature_importance()
```

![shap_summary](https://github.com/user-attachments/assets/c1593885-37c3-4f57-ab47-75a4c9796d51)

![importance](https://github.com/user-attachments/assets/a58f3715-f8a5-4fa6-9a4b-77a5c3170875)

- 주요 영향 변수: `item`, `corporation`, `location`
- `supply(kg)`보다 범주형 요인(거래 주체, 품목)이 결정적  
- 향후 계절성(`timestamp`) 피처 추가 시 예측력 향상 기대

---

## 종합 결론

- 제주 특산물 시장의 가격 변동은 **시계열 요인보다 구조적 요인**(거래 주체·품목·지역)에 의해 더 크게 좌우됨.  
- 공급량이 많을수록 단가가 하락하는 전형적인 수급 메커니즘이 확인되었으며, TG 품목은 고가 프리미엄 군으로 분류.
- 지역 간 가격 차이는 통계적으로 유의하지만 경제적 차이는 미미함.
- 머신러닝 모델은 R² ≈ 0.8 수준으로 가격 예측 안정성을 확보했으며, SHAP 분석을 통해 **“누가, 어떤 품목을, 어디서 거래하느냐”**가 핵심 요인임을 확인.

```
eju_Price_Prediction
 ┣ train.csv
 ┣ jeju_analysis.ipynb
 ┣ 📄 README.md
 ┗ 📄 requirements.txt
```

---

## Author

**윤해정 (Yoon Haejeong)**  
heajeongy@naver.com  
Data Analytics & Visualization / Python, SQL, Power BI  
[GitHub](https://github.com/heajeongy-design)
