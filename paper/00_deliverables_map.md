# 산출물 지도 (Deliverables Map)

**프로젝트:** 정부 R&D 특허의 기술이전(기술료 발생) 예측 — XGBoost + SHAP
**타깃 저널:** Expert Systems with Applications (SCIE, IF ~7.5)
**데이터:** NTIS + KIPRIS 결합, N = 155,522건, 기술료 발생률 26.6%
**작성일 기준:** 2026-07-22

---

## 1. 데이터 자산

| 파일 | 행×열 | 역할 |
|---|---|---|
| `data/analysis_base_v2.parquet` | 155,522 × 23 | **모델링 베이스**. 타깃(`기술료발생`, 0/1) + 특허 수준 특징 |
| `data/df_royalty_v1.parquet` | 73,803 × 27 | 기술료 계약(기술이전) 원장. 타깃 정의 근거·기술료 규모 서술 |

## 2. 단계별 산출물

| 단계 | 내용 | 코드 | 산출물 |
|---|---|---|---|
| 1 | 산출물 지도 | — | `paper/00_deliverables_map.md` |
| 2 | 변수 정의표 + 가설 3개 | `src/s2_variables.py` | `paper/01_variable_hypothesis.md`, `paper/GPT_review_2단계.md` |
| 3 | EDA·기초통계·그룹비교·상관 | `src/s3_eda.py` | `output/tables/*.csv`, `output/figures/eda_*.png` |
| 4 | 모델 비교 (LR·RF·XGB·LGBM) | `src/s4_modeling.py` | `output/tables/model_performance.csv`, `output/figures/roc_pr.png`, `output/models/*.pkl`, `paper/GPT_review_4단계.md` |
| 5 | SHAP 해석 | `src/s5_shap.py` | `output/figures/shap_*.png`, `output/tables/shap_importance.csv` |
| 6 | 강건성 (연도별·기관유형별) | `src/s6_robustness.py` | `output/tables/robustness_*.csv`, `output/figures/robustness_*.png` |
| 7 | 논문 초안 | — | `paper/draft_full.md`, `paper/GPT_review_7단계.md` |

## 3. 진행 로그 / 검증 파일

- `paper/progress_log.md` — 각 단계 완료 시 결과 요약 누적 기록
- `paper/GPT_review_[단계].md` — 2·4·7단계 완료 시 GPT 검증 요청용 요약

## 4. 폴더 구조

```
patent_value/
├── data/            # 원천 parquet (읽기 전용)
├── src/             # 실행 스크립트 (s2~s6)
├── output/
│   ├── tables/      # CSV 표
│   ├── figures/     # PNG 그림
│   └── models/      # 학습된 모델 pkl
└── paper/           # md 산출물 (지도·변수·초안·검증·로그)
```

## 5. 방법론 파이프라인 요약

원천 결합(NTIS+KIPRIS) → 특허 수준 피처 엔지니어링 → 층화 train/test 분할 →
4개 모델 벤치마크(LR·RF·XGBoost·LightGBM, 5-fold CV) → 최적 모델 SHAP 해석 →
연도·기관유형 하위집단 강건성 → 논문화.
