# 제주 특산물 가격 예측 AI 경진대회 (Dacon_Jeju_price_prediction)
https://dacon.io/competitions/official/236176/overview/description

## 프로젝트 개요
본 프로젝트는 **제주 지역 품목별 가격 데이터를 활용한 예측 분석 및 시각화 프로젝트**입니다.  
Python 기반의 통계적 분석(EDA)과 Power BI 시각화를 결합하여, **가격 변동 요인 분석 → 통계 검정 → 머신러닝 예측 → 피처 중요도 해석** 과정을 수행했습니다.

> **프로젝트 목표**
> - 가격(`price(원/kg)`) 변동의 주요 요인 식별  
> - 지역(`location`), 품목(`item`), 법인(`corporation`) 간 구조적 차이 분석  
> - Power BI 대시보드와 Python 모델링 결합으로 시장 구조의 시각적 해석  

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

- 총 데이터 수: **59,397행**
- 결측값: **없음**
- `timestamp` 컬럼은 `datetime64`로 변환 후 시계열 분석에 사용

---

## Power BI 대시보드 구성
<img width="1304" height="728" alt="image" src="https://github.com/user-attachments/assets/6a04ba8e-3e09-4bb0-aa2d-c456c713cf4b" />



| 시각화 유형 | 주요 축/필드 | 분석 목적 |
|--------------|--------------|------------|
| Line Chart | `timestamp` vs `price(원/kg)` | 시계열 가격 추이 |
| Stacked Bar | `location` vs 평균 `price(원/kg)` | 지역별 단가 비교 |
| Pie Chart | `location` vs 총합 `price(원/kg)` | 지역별 점유율 |
| Scatter (Play Axis) | `supply(kg)` vs `price(원/kg)` | 공급-가격 관계 |
| Slicer | `item`, `location`, `corporation`, `timestamp` | 필터 기능 |

> Power BI에서 Python 그래프를 내장 실행하고,  
> `DAX`로 누적 증감률 및 그룹별 KPI 계산식 포함

---

## Python 분석 흐름

### 데이터 로드 및 결측치 처리

```python
df = pd.read_csv("train.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
```

---

### 시계열 가격 추이 분석
<img width="1014" height="547" alt="image" src="https://github.com/user-attachments/assets/73306426-1c82-4aa1-88e5-20427f9d8105" />

- **2019~2023년 평균 단가가 계절적 주기성을 보이며 등락 반복**
- 여름/초기에는 상승, 겨울엔 하락 → **계절성·공급 영향 명확**
- 연도별 완만한 상승 → 물류비, 인건비 상승 등 외부 요인 반영

---

### 지역별 품목 평균 단가 (누적 막대그래프)
<img width="1192" height="590" alt="image" src="https://github.com/user-attachments/assets/f6c8abed-14ad-4f25-a951-6988d78ccfd6" />

- **J 지역:** 평균 약 5,000원/kg, 품목 다양성 높음  
- **S 지역:** 단가 낮고 TG 품목 의존도 높음 → 리스크 집중  
- 지역 간 차이는 단가 수준보다 **품목 구성 다양성 차이**로 해석 가능

---

### Gaussian Mixture Clustering

```python
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

numeric_cols = df.select_dtypes(include=np.number).columns
X_scaled = StandardScaler().fit_transform(df[numeric_cols])
gmm = GaussianMixture(n_components=3, random_state=42)
df['Cluster_GMM'] = gmm.fit_predict(X_scaled)
```

- 확률 기반 군집화로 **데이터의 잠재 구조 탐색**
- 각 군집이 `price(원/kg)`·`supply(kg)`와 어떤 상관이 있는지 추가 분석 가능
- 예측 모델링의 **파생변수로 활용 가능**

---

### OLS 회귀 분석

- **모델식:**  
  `price(원/kg) ~ supply(kg) × item × location × corporation + Cluster_GMM`
- **Adjusted R² = 0.757**
- **유의변수:** `supply(kg)`, `item`, `corporation`, `Cluster_GMM`
- 공급량이 많을수록 가격 하락 / 품목·법인 결합효과 유의  
- 다중공선성 존재 → Ridge/Lasso 정규화 개선 여지 있음

---

### Kruskal–Wallis Test (비모수 검정)

```python
from scipy.stats import kruskal

groups = [g[col].dropna() for _, g in df.groupby('location')]
stat, p = kruskal(*groups)
```

- 지역 간 수치형 변수의 유의 차이 **없음 (p > 0.05)**  
- 지역보다 **품목·공급·법인 구조가 주요 요인**임을 확인  
- 지역 단독 변수보다는 상호작용항 형태로 활용 권장

---

### Boosting + Stacking (머신러닝 예측 모델)

```python
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

stack_model = StackingRegressor(
    estimators=[
        ('lgb', LGBMRegressor(...)),
        ('cat', CatBoostRegressor(...)),
        ('hgb', HistGradientBoostingRegressor(...))
    ],
    final_estimator=XGBRegressor(...),
    n_jobs=-1
)
stack_model.fit(X_train_scaled, y_train)
```

| Metric | Score |
|---------|--------|
| R² | **0.80** |
| RMSE | **안정적 수렴** |
| MAE | **낮은 오차율** |

> Ensemble을 통한 성능 향상 확인 → 단일 모델보다 예측 안정성 우수

---

### Feature Importance (Permutation Importance)
<img width="918" height="547" alt="image" src="https://github.com/user-attachments/assets/1ec26ecc-9eee-4423-b167-6e8a0df5bf30" />

<img width="617" height="393" alt="image" src="https://github.com/user-attachments/assets/d1e324ff-2c61-4caa-a3a4-8dcca36da2e3" />

- **Top Features:**  
  1️ `C(item)[T.RD]`  
  2️ `Intercept`  
  3️ `C(item)[T.CB]`  
  4️ `C(item)[T.CR]`  
- 즉, **품목(item)** 변수가 가장 높은 중요도를 보임  
- 공급량보다 품목 구분이 단가 예측에 결정적 영향

---

## 종합 결론

- 가격 변동은 **시계열 요인보다 구조적 요인**에 더 큰 영향  
- 공급량↑ → 단가↓ : 전형적인 수급 메커니즘 작동  
- 품목·법인·지역이 결합되어 가격을 형성 (특히 TG 품목 프리미엄)  
- 지역 간 평균 가격 차이는 **통계적으로 유의하나, 경제적 크기는 작음**  
- 머신러닝 결과, `timestamp`·`supply(kg)`보다 **거래 주체·품목 특성**의 영향이 큼  

> **결론:**  
이번 프로젝트는 단순한 가격 예측을 넘어, 데이터의 구조적 패턴을 이해하고 시장의 메커니즘을 분석한 사례로 평가할 수 있다.
EDA와 통계 분석을 통해 가격의 단기 변동성이 주로 공급량(supply) 및 계절적 요인에 의해 영향을 받는 반면, 장기적으로는 품목(item) 및 법인(corporation) 의 구조적 차이가 더 큰 설명력을 가진다는 점을 확인했다.

특히 Gaussian Mixture 기반의 군집화와 OLS 회귀 분석을 결합하여, 단가 형성에 작용하는 다층적 관계(품목×법인×지역×공급) 를 정량적으로 검증했다는 점이 본 프로젝트의 차별화된 부분이다.
또한, 비모수 검정(Kruskal–Wallis)을 통해 지역 간 단순 평균 차이가 유의하지 않음을 검증함으로써, 지역적 요인보다는 시장 구조적 특성이 가격 결정의 핵심임을 데이터로 입증하였다.

머신러닝 단계에서는 LightGBM, CatBoost, XGBoost 등을 결합한 Stacking Ensemble 모델을 구축하여, 단일 알고리즘 대비 높은 안정성과 예측력을 확보했다.
Permutation Importance 결과 또한 ‘품목(item)’의 영향력이 가장 높게 나타났으며, 이는 시장 내 상품 구성의 다양성 및 프리미엄 구조가 가격 변동의 중심에 있음을 시사한다.

결론적으로, 본 분석은 제주 특산물 시장의 가격 형성이 단순한 시계열적 패턴이 아닌 공급·품목·법인 간 복합 상호작용의 결과임을 보여준다.
향후에는 계절성 변수(월, 분기), 날씨, 수요 탄력성 등을 포함한 시계열 예측모델(Time Series Forecasting) 로 확장함으로써, 더 정교한 가격 예측과 실무적 활용이 가능할 것으로 기대된다.


---

## 사용 기술 스택

| 구분 | 기술 |
|------|------|
| 데이터 처리 | Python (pandas, numpy, scipy, statsmodels) |
| 모델링 | scikit-learn, XGBoost, LightGBM, CatBoost |
| 시각화 | seaborn, matplotlib, Power BI |
| 통계검정 | OLS, Kruskal-Wallis, ANOVA |
| 클러스터링 | Gaussian Mixture Model |
| 환경 | Google Colab + Power BI Desktop |

---

## Repository Structure

```
eju_Price_Prediction
 ┣ 📄 train.csv
 ┣ 📄 jeju_analysis.ipynb
 ┣ 📄 PowerBI_Dashboard.pbix
 ┣ 📄 README.md
 ┗ 📄 requirements.txt
```

---

## 👤 Author

**윤해정 (Yoon Haejeong)**  
heajeongy@naver.com  
Data Analytics & Visualization / Python, SQL, Power BI  
[GitHub](https://github.com/heajeongy-design)
