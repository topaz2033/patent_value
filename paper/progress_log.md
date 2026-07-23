# 진행 로그 (Progress Log)

**프로젝트:** 정부 R&D 특허의 기술이전(기술료 발생) 예측 — XGBoost + SHAP
**최종 갱신:** 2026-07-23

---

## 1단계 — 산출물 지도 ✅
- `paper/00_deliverables_map.md` 작성. 데이터 자산·단계별 산출물·폴더 구조 정의.

## 2단계 — 변수 정의 + 가설 ✅
- 코드: `src/s2_variables.py`, 공통: `src/common.py`
- 산출: `output/tables/variable_summary_{numeric,categorical}.csv`, `paper/01_variable_hypothesis.md`
- 핵심: 타깃 발생률 26.6%. 개별 특허 품질과 타깃의 선형 상관 약함(최대 r≈0.044), 맥락 변수 집단 격차 큼 → H1–H3 도출.

## 3단계 — EDA ✅
- 코드: `src/s3_eda.py`
- 산출: `eda_group_compare_{numeric,categorical}.csv`, `eda_rate_by_*.csv`(4), 그림 5개.
- 핵심 결과:
  - 수치형 발생/비발생 차이 모두 **효과크기 작음**. 최대 `claim_total` Cohen's d=0.099(유의), `inventor_count` d=0.049. `ipc_count`·`backward_citation`·`backward_foreign`·`특허연령`은 유의하지 않거나 무시할 수준.
  - 범주형 χ²: **부처유형 Cramér's V=0.130**(최대), 기관유형 0.065(둘 다 p<1e-100). `finalDisposal`·`registerStatus`는 유의하지 않음.
  - 집단 발생률: 기관유형 출연연 29.4% > 대학 27.9% > 기업 23.8% > 기타 17.7%. 부처유형 9.6~35.2%.

## 4단계 — 모델 벤치마크 ✅
- 코드: `src/s4_modeling.py`
- 산출: `model_performance.csv`, 그림 `roc_pr.png`, 모델 pkl 4종, `models/_meta.json`, `test_index.npy`
- 층화 분할 train 124,417 / test 31,105, `scale_pos_weight`=2.757, 5-fold CV.
- Test 성능 (ROC-AUC 순):

  | 모델 | CV AUC | Test AUC | Test PR-AUC | F1 | Brier |
  |---|---|---|---|---|---|
  | RandomForest | 0.7035±0.0010 | **0.7104** | 0.4621 | 0.4706 | 0.1984 |
  | LightGBM | 0.7007±0.0017 | 0.7066 | 0.4595 | 0.4954 | 0.2133 |
  | XGBoost | 0.6997±0.0024 | 0.7045 | 0.4554 | 0.4939 | 0.2155 |
  | LogisticRegression | 0.6058±0.0025 | 0.6020 | 0.3400 | 0.4165 | 0.2418 |

  - 트리 3종은 통계적으로 동률(ΔAUC<0.006), 선형 모델 대비 +0.10 AUC → **비선형·상호작용 구조** 확인(H도출 근거 지지).
  - SHAP 해석 대상은 사전 선언 방법론(README: XGBoost+SHAP)·문헌 표준에 따라 **XGBoost**. RF의 근소 우위는 검증 문서에 명시.

## 5단계 — SHAP 해석 ✅
- 코드: `src/s5_shap.py`
- 산출: `shap_importance.csv`, 그림 3개(beeswarm·bar·dependence), `models/_shap_summary.json`
- 전역 중요도(mean |SHAP|, 기여도%):
  1. 등록연도 21.2% · 2. 특허연령(월) 8.2% · 3. 부처유형=교육부계 7.1% · 4. 기관유형=대학 6.2% · 5. 부처유형=과기부계 6.0% · 6. 출원일자결측 5.3% · 7. 독립청구항 4.5% · 8. 청구항 4.5% · 9. 발명자 수 4.2% · 10. 기관유형=출연연 3.6%
  - **시간·행정 변수(등록연도·특허연령)가 최상위** → 관측창/중도절단 영향 시사. **정책·기관 맥락(부처·기관유형)이 그다음 tier**로 개별 품질 지표 대부분보다 상위 → H3 지지.

## 6단계 — 강건성 ✅
- 코드: `src/s6_robustness.py`
- 산출: `robustness_by_year.csv`, `robustness_by_institution.csv`, `robustness_out_of_time.csv`, 그림 2개
- 하위집단: 연도별 AUC 0.667~0.726, 기관유형별 0.672~0.752 (전체 0.7045) → **하위집단 안정적**.
- 시계열 외삽(out-of-time): ≤2019→0.598, ≤2020→0.597, ≤2021→0.614 → 무작위분할 대비 **AUC ~0.10 하락**. 시간 분포 변화 존재(정직 보고).

## 7단계(민감도) — 등록연도 제외 재분석 ✅
- 코드: `src/s7_sensitivity.py`
- 산출: `sensitivity_registration_year.csv`, `shap_importance_noyear.csv`, 그림 `sensitivity_registration_year.png`
- 동일 분할(seed 42)로 등록연도 제거 후 재학습:

  | 모델 | AUC 원본 | AUC 등록연도제외 | ΔAUC |
  |---|---|---|---|
  | LR | 0.6020 | 0.6021 | +0.000 |
  | RF | 0.7104 | 0.6521 | −0.058 |
  | XGB | 0.7045 | 0.6478 | −0.057 |
  | LGBM | 0.7066 | 0.6466 | −0.060 |

  - 트리 AUC ~0.06 하락하나 **0.65 유지** → 등록연도는 실질 신호를 담되 예측이 관측창 아티팩트만은 아님.
  - 등록연도 제외 SHAP 상위: 부처유형=교육부계 10.0% · 특허연령 8.2% · 부처유형=과기부계 7.9% · 출원일자결측 6.8% · 청구항 6.1% · 기관유형(출연연 5.5%·대학 4.7%) → **정책·기관 맥락 부상, H3 강화**.

## 7단계 — 논문 초안 ✅
- 산출: `paper/draft_full.md`, `paper/GPT_review_4단계.md`, `paper/GPT_review_7단계.md`
- 전 단계 결과 통합, H1–H3 검정 결론 및 한계(시계열 일반화·중도절단) 명시.
