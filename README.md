# 제주 특산물 가격 예측 AI & Power BI 통합 분석
https://dacon.io/competitions/official/236176/overview/description

## 프로젝트 개요
이 프로젝트는 제주 지역 특산물 시장의 공급량(Supply)과 가격(Price) 을 예측하고,  
Power BI 대시보드를 통해 시장 구조를 시각적으로 분석하기 위한 데이터 분석 프로젝트입니다.  

Python 기반의 통계 분석과 머신러닝 모델링,  
Power BI 시각화를 결합하여 예측력 향상과 인사이트 도출을 함께 수행했습니다.

> 목적:  
> - 가격(`price(원/kg)`) 변동의 주요 요인 식별  
> - 지역(`location`), 품목(`item`), 법인(`corporation`)간의 구조적 차이 파악  
> - Power BI 대시보드로 시장 구조를 직관적으로 표현

---

## 데이터 개요

| 컬럼명 | 설명 | 타입 |
|--------|------|------|
| ID | 거래 고유 식별자 | object |
| timestamp | 거래 발생 시점 | datetime |
| item | 품목명 (예: BC, CB, CR, RD, TG) | object |
| corporation | 거래 주체 | object |
| location | 거래 지역 (J: 제주 / S: 서울) | object |
| supply(kg) | 공급량 (kg 단위) | float |
| price(원/kg) | 거래 단가 (원/kg) | float |

데이터 크기: 59,397행 × 7열  
기간: 2019년 ~ 2023년  
결측치: 없음

---

## 분석 및 모델링 절차

| 단계 | 내용 | 목적 |
|------|------|------|
| 1 | 데이터 전처리 (결측치 처리, 이상치 제거) | 분석 신뢰성 확보 |
| 2 | EDA (탐색적 데이터 분석) | 공급량·가격 패턴 및 주요 변수 탐색 |
| 3 | 다항회귀 (Polynomial Regression, degree=2) | 비선형 관계 학습 |
| 4 | ANOVA 검정 | 변수별 평균 차이 검정 (통계적 유의성 확인) |
| 5 | 모델 학습 | ElasticNet / RandomForest / GradientBoosting |
| 6 | 스태킹 앙상블 | Boosting + Ridge Meta Model 결합 |
| 7 | Feature Importance | 주요 변수 영향도 분석 |
| 8 | Power BI 시각화 통합 | 예측결과 및 구조 시각화 |

---

## 모델 구성 (Python 기반)

```python
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor

stack_model = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
    ],
    final_estimator=RidgeCV(),
    n_jobs=-1
)
```

### 모델 성능 요약

| 모델 | R² | RMSE |
|------|----|------|
| ElasticNet | 0.42 | 2.40 |
| RandomForest | 0.73 | 1.89 |
| GradientBoosting | 0.76 | 1.82 |
| Stacking (최종) | 0.82 | 1.74 |

---

## Feature Importance 분석

### GradientBoosting 기준 상위 중요 변수

| 순위 | 변수명 | 설명 |
|------|--------|------|
| 1 | supply(kg) | 공급량이 가격 결정의 핵심 변수 |
| 2 | location | 지역별 시장 가격 차이 |
| 3 | item | 품목별 단가 차이 |
| 4 | corporation | 공급 주체의 영향 |
| 5 | timestamp | 시즌별 시장 변동 반영 |

```python
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
```

---

## Power BI 시각화 통합
<img width="1294" height="733" alt="image" src="https://github.com/user-attachments/assets/50bfbf8a-3109-4d79-a56c-bbae01e73ab3" />


### 대시보드 개요
**제주 특산물 가격 예측을 위한 시장 구조 파악**



| 구성 요소 | 설명 |
|------------|------|
| 연도별 Item 단가 변화 | 연도별 평균 단가 추이 (2019~2023) |
| 지역별 Item 단가 비교 | 제주(J) vs 서울(S) 시장 단가 차이 |
| Supply별 단가 동적 관계 | 공급량과 단가의 시계열 상관관계 |
| 지역별 시장 점유율 | 공급량 기준 시장 점유 비율 |
| 상단 필터 | Location / Supply별 조건부 분석 기능 |

---

## 프로젝트 주요 인사이트

| 구분 | 요약 |
|------|------|
| 시장 구조 | 서울 지역의 평균 단가가 제주보다 높게 형성됨 |
| 공급량-가격 관계 | 공급량이 증가할수록 단가 하락 경향 |
| 연도별 트렌드 | 2022년 이후 전반적 가격 하락세 |
| 예측 모델 | 스태킹을 통한 예측력(R²) 0.82 달성 |
| 시각화 통합 | BI 대시보드로 시장 구조 및 가격 동향 한눈에 확인 가능 |

---

## 폴더 구조

```
제주_특산물_가격_예측_AI/
├── README.md
├── PowerBI_Dashboard.pbix
├── 제주_특산물_가격_예측_AI_(해정).ipynb
├── data/
│   ├── raw_data.csv
│   ├── processed_data.csv
│   └── feature_importance.csv
└── output/
    ├── model_predictions.csv
    ├── importance_chart.png
    └── eclo_dashboard_export.csv
```

---

## 기술 스택

| 구분 | 사용 기술 |
|------|------------|
| 분석 언어 | Python (Pandas, NumPy, scikit-learn) |
| 모델링 | ElasticNet, RandomForest, GradientBoosting, Stacking |
| 시각화 | Power BI, Seaborn, Matplotlib |
| 통계 분석 | ANOVA, Permutation Importance |
| 환경 | Google Colab, Power BI Desktop |

---

## 결론 및 향후 발전 방향

1. Power BI 자동 업데이트 파이프라인 구축 (Python → CSV → BI)
2. 가격 구간 분류(Classification) 접근으로 예측 해석 강화
3. SHAP, Permutation Importance 기반 영향도 분석 추가
4. 계절별, 월별 시계열 트렌드 강화 분석

---

## 프로젝트 요약 문장
Python 기반 머신러닝 모델로 제주 특산물 가격을 예측하고,  
Power BI 대시보드를 통해 시장 구조와 가격 변동 요인을  
시각적으로 분석한 데이터 통합 프로젝트.

---

© 2025. Data & BI Project by 윤해정

