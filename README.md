# Patent Value — 정부 R&D 특허의 기술료 발생 예측

NTIS 성과자료와 KIPRIS 특허 서지정보를 활용해 기술료 발생 여부를 분류하고, 모델별 성능과 예측 요인을 비교하는 연구 코드 저장소입니다. 기술이전·연구성과 활용을 데이터로 검토하는 분석 사례를 담고 있습니다.

**현재 상태:** 탐색적 연구 코드와 논문 초안입니다. 아래 수치는 저장소에 포함된 기존 결과표 기준이며, 이번 문서 정리 과정에서 원시 데이터로 재실행한 결과가 아닙니다. 논문 게재·심사 완료나 실제 업무 적용 성능을 의미하지 않습니다.

## 분석 범위

- 관측 단위: 등록특허 1건. 저장된 데이터 설명 기준 155,522건, 기술료 발생 41,391건(26.6%).
- 목표: 관측기간 내 기술료 발생 여부의 분류. 기술의 금전적 가치 자체를 추정하는 모형과는 구분합니다.
- 구현: 변수 정의·탐색적 분석 → LR·Random Forest·XGBoost·LightGBM 비교 → XGBoost SHAP 해석 → 하위집단·시계열 검증 → 등록연도 제외 민감도 분석.
- 공개 코드의 시작점: 가공된 parquet 파일입니다. 원천자료 수집부터 특허번호 결합·타깃 생성까지의 전처리 전체는 이 저장소에 포함되어 있지 않습니다.

## 저장된 주요 결과

[모델 비교 결과표](output/tables/model_performance.csv)의 무작위 층화 분할 평가입니다.

| 모델 | Test ROC-AUC | Average precision (CSV의 Test_PR_AUC) |
|---|---:|---:|
| Random Forest | 0.7104 | 0.4621 |
| LightGBM | 0.7066 | 0.4595 |
| XGBoost | 0.7045 | 0.4554 |
| Logistic Regression | 0.6020 | 0.3400 |

저장된 비교표에서 ROC-AUC가 가장 높은 모델은 Random Forest이며, SHAP 해석과 하위집단 검증 코드는 XGBoost를 사용합니다. 모델 간 차이의 통계적 유의성을 이 표만으로 판단하지 않습니다.

XGBoost의 [시계열 외삽 결과](output/tables/robustness_out_of_time.csv)는 ROC-AUC 0.5966~0.6135로, 무작위 분할 결과보다 낮습니다. [등록연도 제외 민감도 분석](output/tables/sensitivity_registration_year.csv)도 함께 확인해야 합니다.

## 해석과 검증의 한계

- SHAP은 모델의 예측 기여를 설명하며, 기관 유형이나 정책의 인과적 효과를 입증하지 않습니다.
- 기술료 발생 여부의 관측기간과 변수 수집 시점이 실제 예측 시점에 맞는지 추가 검증이 필요합니다. 현재 결과를 미래 기술이전의 확정적 예측으로 해석하지 않습니다.
- 현재 코드는 분할 전에 전체 자료의 중앙값으로 일부 결측치를 대체합니다. 시계열 검증의 클래스 가중치도 전체 타깃으로 계산합니다. 학습 자료 안에서만 전처리와 가중치를 추정하도록 보완한 뒤 성능을 재평가해야 합니다.
- 원시 자료와 데이터 결합 과정이 공개되어 있지 않아, 공개 파일만으로 수치의 독립적 재현을 완료할 수 없습니다.
- Python·패키지 버전은 기존 환경 기록입니다. 새 환경에서의 설치와 전체 실행은 별도로 확인해야 합니다.

## 살펴볼 파일

| 목적 | 파일 |
|---|---|
| 변수·타깃 정의 | [paper/01_variable_hypothesis.md](paper/01_variable_hypothesis.md) |
| 모델 비교 구현 | [src/s4_modeling.py](src/s4_modeling.py) |
| SHAP 해석 | [src/s5_shap.py](src/s5_shap.py) |
| 시계열·하위집단 검증 | [src/s6_robustness.py](src/s6_robustness.py) |
| 민감도 분석 | [src/s7_sensitivity.py](src/s7_sensitivity.py) |
| 결과표·그림 | [output/](output/) |
| 논문 작업 초안 | [paper/draft_full_v2.md](paper/draft_full_v2.md) |

`paper/`에는 초안과 과거 검토 요청·진행 기록이 함께 있습니다. 과거 완료 표시나 검토 요청은 독립적인 검증 또는 게재 확정을 뜻하지 않습니다.

## 데이터 준비와 실행

원시 데이터 파일은 저장소에 포함하지 않습니다. 적법하게 이용할 수 있는 자료를 확보하고 [data/README.md](data/README.md)의 설명에 따라 다음 가공 파일을 준비해야 합니다.

- `data/analysis_base_v2.parquet`
- `data/df_royalty_v1.parquet`

기존 기록의 Python 버전은 3.12.10이며 패키지 목록은 [requirements.txt](requirements.txt)에 있습니다.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_all.py
Set-Location src
python s7_sensitivity.py
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
cd src
python s7_sensitivity.py
```

`run_all.py`는 s2~s6만 실행합니다. s7 민감도 분석은 위와 같이 별도로 실행합니다. s5·s6는 s4가 생성하는 `XGBoost.pkl`과 `test_index.npy`에 의존합니다. 실행하면 기존 `output/` 결과 파일이 덮어써질 수 있으므로 비교할 결과는 먼저 보관하세요.

## 공개 범위와 이용 조건

- 코드 이용 조건은 [LICENSE.md](LICENSE.md)를 따릅니다.
- 원시 parquet와 학습 모델 바이너리는 배포 대상에서 제외합니다.
- 공개 스키마 파일은 컬럼명·자료형·결측 수·고유값 수만 제공하며 개별 값 샘플은 제거했습니다.
- 현재 파일에서 샘플을 제거해도 이전 커밋의 내용까지 삭제되지는 않습니다.
- 원천 데이터의 재배포 가능 여부와 논문 초안의 공동저자·출판 관련 권한은 별도로 확인해야 합니다.
