# 미국 대도시 보건 데이터셋을 기반으로 한 질병 발병 및 사망 통계 예측 AI 모델

**프로젝트 구성원** : 김남덕, 김명윤, 신수웅, 오도은, 최재원 / [**발표 slide 자료**](https://docs.google.com/presentation/d/1LSaatqx-LFgNHZMdRt6-Qej-CfJq610s0M-9tymdhk0/edit?usp=sharing)


## 1. 프로젝트 개요

### 프로젝트 배경
- 질병발생 예측 연구는 미래 질병 관리를 위한 중요한 분야
- 팀원들의 도메인 지식과 관련된 분야로, 정형 데이터를 활용한 분석 프로젝트

## 2. 목적
- 미국 대도시의 생활 환경 데이터를 기반으로 주요 질병의 발병률 및 사망률을 예측하는 모델을 개발하여 예방의학 발전을 도모
- Random Forest, k-NN, XGBoost 및 Multi-Layer Perceptron(MLP) 모델을 이용하여 예측
- 적절한 성능지표를 이용하여 회귀 예측에 적합한 모델링 개발


## 3. 데이터셋
### 1) 데이터 개요
- [Big Cities Health Inventory(BCHI)](https://bigcitieshealthdata.org/)
- 미국 대도시들에 대해 건강, 기후 및 환경, 경제적 불평등 등 건강 지표 및 건강 지표에 영향 줄 수 있는 다양한 통계가 집계되어 있음


### 2) 데이터 셋 구조
- 미국 전역 및 35개 대도시에 대한 2010~2022년의 통계 자료,
총 189,979건의 레코드로 구성됨됨

![df](./imgs/1.df.png)
_그림 1. BCHI 데이터 셋_

- 각 레코드는 층화된 집단에 대한 특정 지역 및 연도의 통계와 관련된 내용
   - [층화된 집단에 대한 정보, 통계 종류 및 분류, 통계값, 단위, 신뢰구간, 데이터의 출처 등으로 구성](./research/240614_step0.ipynb)
   - 인종, 성별, 도시의 특성을 기준으로 층화시켜 조사한 통계 자료
   - **인종** : White, Black, Hispanic, Asian/PI, Natives 및 All (i.e. 인종에 대해 층화되지 않음)
   - **성별** : Female, Male 및 Both (i.e. 성별에 대해 층화되지 않음)
   - **도시의 특성** : 지역, 경제적 풍요, 거주지에서 인종 분리 정도, 인구, 인구밀도
        (도시 특성 중 지역 외에는 모두 binary ordinal / 모두 종합하면 19개의 도시 유형이 나옴)
   - 통계 자료의 종류는 총 118종의 지표 (모두 numeric)


### 3) EDA
- 레코드의 특성 관련한 칼럼에 대하여, [신뢰 구간](./research/240617_ciEDA.ipynb), [인구 및 성별 층화 관련 결측](./research/240619_check_missing_entire.ipynb), [지리적 정보에 관한 칼럼](./research/240619_EDA_geo.ipynb), [통계값의 단위와 스케일 및 분포 간의 관계](./research/240619_variance_feature.ipynb) 등에 대하여 조사
- 위의 조사들을 통해 데이터 분포의 특성 및 결측값의 분포 등에 대하여 파악함
- 인종, 성별 층화를 고려하면 결측률이 낮은 통계지표는 10여개에 불과하지만, 참고할 만한 최소한의 데이터는 많은 수의 통계지표가 가지고 있는 것을 확인
- 신뢰 구간 관련 조사하는 과정에서, 데이터셋에 포함된 대다수의 통계 지표가 연도보다 도시의 특성에 따라 값의 분포가 변화하는 것을 $\chi^2$ 검정을 통해 확인

## 3. 문제 설정


**목표** : 도시의 특성/인종/성별로 층화된 인구집단에 대하여, 관련있는 여러 통계 자료를 바탕으로 특정 질병의 발병 및 사망에 관한 통계를 예측하고자 함

### 1) 종속 변수 설정
**기준** : 결측치를 채울 수 없는 점을 고려하여, 결측률이 낮고 주요한 질병/사망요인에 대한 통계지표를 종속변수로 설정

|분류|항목|
|-------|------------------|
|Cancer | All Cancer Deaths|
|Cancer | Colorectal Cancer Deaths|
|Cancer | Lung Cancer Deaths|
|Cardiovascular Disease | Cardiovascular Disease Deaths|
|Cardiovascular Disease | Heart Disease Deaths|
|Deaths | Deaths from All Causes|
|Deaths | Gun Deaths (Firearms)|
|Deaths | Injury Deaths|
|Deaths | Motor Vehicle Deaths|
|Deaths | Premature Death|
|Diabetes and Obesity | Diabetes Deaths|
|Life Expectancy at Birth | Life Expectancy|
|Mental Health | Suicide|
|Substance Use | Drug Overdose Death|

_표 1. 사용된 종속 변수 목록_


### 2) 독립 변수 설정
- 각 종속변수 별로 별개의 독립변수를 선정
- 종속변수를 제외한 지표 중에서 독립변수 선정
- 도메인 지식을 활용하여 종속변수 별로 해당하는 Feature 선별
- [통계적 접근으로 독립변수로 활용하면 안될 Feature 선별](./research/240619_indvar.ipynb)
- [종속변수별로 결과를 직접 드러낼 수 있는 일부 Feature 제거](./research/set_lists.ipynb)

![indvar](./imgs/2.varlist.png)
_그림 2. 각 종속변수에 따라 설정된 독립 변수 후보의 예시_

## 4. 전처리
**전체 과정**
- raw data를 pivot table로 변형, k-NN 모델로 결측치 보간, 독립변수를 대상으로 scaling 진행, nominal 데이터에 대한 encoding 등
- 도시,연도,인종,성별로 층화된 각 표본을 row로, 각 표본의 층화 정보와 118개 통계 지표를 column으로 한 pivot table로 변형

![pvtb](./imgs/3.pvtb.png)
_그림 3. pivot table 변형 후 데이터_

**결측치 보간**
- 분포의 형태 및 집계 데이터임을 감안하여 이상치 기준은 설정하지 않음
- 세부적으로 층화된 샘플 집단에 대해 결측치를 해결하는 것이 주요한 과제
- 각 통계 지표에 대해서 인구/성별/도시의 특성에 따라 층화된 정보를 바탕으로 통계치를 예측하는 모델을 만들어, 결측치를 보간하고자 함
  - 가장 가까운 집단에서의 값을 참고한다는 직관에 따라 [k-NN regressor](./research/240620_how_to_fill_missing_knn.ipynb) 사용
  - 각 층화 특성에서의 거리에 weight를 반영된 custom metric을 구현
  - Euclidean 혹은 weight가 반영되지 않은 custom metric에 비해 유의미하게 좋은 성능을 보임
  - weight의 값은 도메인 지식과 EDA 결과를 바탕으로 휴리스틱하게 결정
  - 하지만 custom metric의 경우 최적화가 덜 되어 train 및 predict에서 걸리는 시간이 통계 지표 하나당 분 단위로 걸리는 단점이 있음
  - [Decision Tree](./research/240620_how_to_fill_missing_with_dt.ipynb)를 이용한 모델도 구현해본 결과, k-NN에 준하는 성능을 얻음 

![kNN관련결과](./imgs/4.knn.png)
_그림 4. train셋의 평균을 baseline으로 하였을 때, k-NN regressor 적용 결과 분석 예시 (실제 및 예측값 분포/오차의 분포/성능 지표)_

## 5. 모델
### 1) 모델 선택 및 학습
- ```sklearn.train_test_split```을 사용하여 연도를 기준으로 층화해 train 80%, test 20%로 [분리](./model/data_prep.ipynb)
- [Random Forest](./model/random_forest.ipynb), [XGBoost](./model/boost.ipynb), [MLP](./model/mlp.ipynb) 모델을 사용
- 각 종속변수별로 앞서 정한 독립변수 후보만 사용하는 것과 기타 통계지표도 활용하여 예측하는 것 사이 비교 평가 진행
- grid search를 통해 각 모델별로 가장 높은 성능을 내는 hyper parameter 탐색

![RF학습](./imgs/5.rf.png) _그림 5. 앞서 정한 후보를 대상으로 독립변수를 좁히는 것 여부에 따른 Best Random Foreset Model의 $R^2$ score 성능 비교_

![XGB학습](./imgs/6.xgb.png) _그림 6. 앞서 정한 후보를 대상으로 독립변수를 좁히는 것 여부에 따른 Best XGBoost Model의 $R^2$ score 성능 비교_


### 2) [결과 평가](./model/compare_results.ipynb)
- 모델 성능은 RMSE, MAPE, $R^2$ score 등을 활용하여 평가
- XGBoost 모델과 전처리 과정에서 개발한 k-NN 모델이 최종적으로 가장 우수한 성능을 보였음
  - k-NN 모델이 성능 지표 면에서는 가장 뛰어났음
  - XGBoost 모델은 성능은 k-NN에 준하면서 train/predict에 걸리는 시간이 k-NN 모델에 비해 압도적으로 짧게 걸렸음
- k-NN 모델의 전처리를 이용한 RFC, XGBoost의 모델 학습이 이용하지 않은 것에 비해 성능이 약소하게 높았음
  - XGB의 경우 대개 k-NN 전처리 사용 여부가 성능에 큰 차이는 없었음
  - 하지만 특정 종속변수에서는 큰 차이가 나는 동시에 MAPE와 $R^2$ score 사이에 상반된 결과를 보였음

![결과비교](./imgs/7.rslt.png)
_그림 7-1. Best Model 간의 성능 비교_
![결과비교](./imgs/7.rslt2.png)
_그림 7-2. k-NN, k-NN 전처리를 사용한 XGBoost, 사용하지 않은 XGBoost 간의 성능 비교_

## 6. 분석 결과 및 해석
- 인종, 성별, 도시 관련 특성에 대한 정보로만 학습한 k-NN의 성능이 좋았던 것과 Random Forest의 Feature Importance 분석 결과를 바탕으로 하여, 보건 통계를 예측할 때 인종, 성별이 주요한 역할을 하는 것을 확인 
- 층화된 집단 별로 여러 질병의 발병률과 사망률에 대한 예측 결과는 보건 정책 수립과 자원 배분에 중요한 자료로 활용될 수 있음
- RMSE, MAPE, $R^2$ score 등 지표의 공통적인 경향 및 각 지표의 차이점에 대해 생각해볼 수 있었음

## 7. 결론 및 향후 연구
- 인종, 성별에 대한 정보가 보건 데이터 분석 및 예측에서 매우 주요한 정보임을 확인함
- 머신러닝 모델이 보건 데이터 분석 및 예측에 효과적임을 확인할 수 있었음
- 각 통계 자료에 대하여, 분포의 특징을 보다 세부적으로 고려해 하이퍼 파라미터를 조절하여 k-NN 모델을 개선할 수 있음
- 개선된 k-NN 전처리 결과를 바탕으로 

### References
1. 한국인 유전체분석사업을 통한 한국인 유전체변이 정보 기반의 질병 발생 위험도 예측 모형 고도화 연구- 당뇨병 예측모형을 중심으로-, 질병관리청, 2024
2. https://bigcitieshealthdata.org/
3. Harnessing multimodal data integration to advance precision oncology, Nature Reviews Cancer, 22, 2022, 114-126 
4. Pulungan, A. F., Zarlis, M., & Suwilo, S. (2019). Analysis of Braycurtis, Canberra and Euclidean Distance in KNN Algorithm. Sinkron : Jurnal Dan Penelitian Teknik Informatika, 4(1), 74-77.