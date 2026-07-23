# Patent Value — 정부 R&D 특허의 기술이전(기술료 발생) 예측

정부 R&D 등록특허가 기술이전(기술료 발생)으로 이어질지를 예측하고, SHAP으로 예측 요인을 해석하는 프로젝트.

- **데이터:** NTIS + KIPRIS 결합, N = 155,522건, 기술료 발생률 26.6%
- **방법:** LR·RF·XGBoost·LightGBM 벤치마크 → XGBoost + SHAP 해석 → 강건성·민감도 검증
- **타깃 저널:** Expert Systems with Applications (SCIE)

## 환경

- Python **3.12.10**
- 주요 패키지 (전체·고정 버전은 `requirements.txt`)
  - pandas 3.0.3 · numpy 2.4.6 · scipy 1.18.0 · scikit-learn 1.9.0
  - xgboost 3.3.0 · lightgbm 4.7.0 · shap 0.52.0 · matplotlib 3.11.1 · seaborn 0.13.2

```bash
python -m venv venv
source venv/Scripts/activate      # Windows PowerShell: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 데이터 준비

원시 데이터는 재배포 권한·용량 문제로 저장소에 포함하지 않는다. `data/README.md`를 참고해
`data/analysis_base_v2.parquet`, `data/df_royalty_v1.parquet`를 직접 배치한다.

## 실행

```bash
python run_all.py          # src/s2~s6 순차 실행 → output/ 생성
```

개별 단계 실행(모듈 임포트 때문에 `src/`에서 실행):

```bash
cd src
python s2_variables.py     # 변수 정의 기술통계
python s3_eda.py           # EDA (집단비교·상관·그림)
python s4_modeling.py      # 모델 벤치마크 → output/models/*.pkl, model_performance.csv
python s5_shap.py          # SHAP 해석 (s4의 XGBoost.pkl 필요)
python s6_robustness.py    # 하위집단·시계열 외삽 강건성
python s7_sensitivity.py   # 등록연도 제외 민감도 분석
```

> 재생성 순서 주의: `s5`·`s6`는 `s4`가 저장한 `output/models/XGBoost.pkl`과 `test_index.npy`에 의존한다.
> 반드시 `s4` → `s5`/`s6` 순으로 실행하거나 `run_all.py`를 사용한다.

## 프로젝트 구조

```
patent_value/
├── data/            # 원시 parquet (Git 미포함 — data/README.md 참조)
├── src/
│   ├── common.py            # 경로·폰트·피처 정의·데이터 로드
│   └── s2~s7_*.py           # 단계별 실행 스크립트
├── output/
│   ├── tables/      # CSV 결과표 (변수요약·EDA·성능·SHAP·강건성·민감도)
│   ├── figures/     # PNG 그림
│   └── models/      # 학습 모델(.pkl, Git 미포함) + _meta.json·_shap_summary.json
├── paper/           # 논문 산출물 (초안·변수가설·관련연구·진행로그·검증요청)
│   ├── draft_full.md            # 논문 초안 (국·영 초록 포함, 참고문헌 21편 Crossref 검증)
│   ├── 00_deliverables_map.md   # 산출물 지도
│   ├── 01_variable_hypothesis.md# 변수 정의·가설 H1~H3
│   └── progress_log.md          # 단계별 진행 기록
├── requirements.txt
├── run_all.py
└── .gitignore
```

## 결과 재생성 순서 요약

1. `data/`에 parquet 2종 배치
2. `pip install -r requirements.txt`
3. `python run_all.py` (또는 `src/s2 → s7` 순차 실행)
4. `output/tables`, `output/figures`, `output/models` 생성 확인
5. 논문 초안은 `paper/draft_full.md`
