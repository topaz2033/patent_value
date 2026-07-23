# -*- coding: utf-8 -*-
"""5단계: SHAP 해석 — XGBoost 주 모델.
성능상 XGB·LGBM·RF는 통계적으로 동률(ΔAUC<0.006)이며, 사전 선언된 방법론
(README: 'XGBoost + SHAP')과 기술이전 예측 문헌 표준에 따라 XGBoost를 해석 대상으로 사용.
"""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from common import load_base, build_xy, LABELS, TAB, FIG, MOD, SEED, savefig

# ---------- 모델 · 데이터 복원 ----------
with open(MOD / "XGBoost.pkl", "rb") as f:
    model = pickle.load(f)
df = load_base()
X, y = build_xy(df)
test_idx = np.load(MOD / "test_index.npy")
X_te = X.loc[test_idx]

# 사람이 읽는 컬럼 라벨 (원-핫은 접두어 유지)
def pretty(col):
    if col in LABELS:
        return LABELS[col]
    for base in ("기관유형_", "부처유형_", "finalDisposal_", "registerStatus_"):
        if col.startswith(base):
            return f"{LABELS.get(base[:-1], base[:-1])}={col[len(base):]}"
    return col

feat_labels = [pretty(c) for c in X_te.columns]

# ---------- SHAP 값 (TreeExplainer, 전체 test) ----------
print(f"[s5] SHAP on XGBoost, test N={len(X_te):,}, features={X_te.shape[1]}")
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_te)          # (N, F)
print("[s5] shap_values shape:", np.asarray(sv).shape)

# ---------- (1) 중요도 표 (mean |SHAP|) ----------
mean_abs = np.abs(sv).mean(axis=0)
imp = (pd.DataFrame({"변수": X_te.columns, "라벨": feat_labels,
                     "mean_abs_SHAP": mean_abs})
       .sort_values("mean_abs_SHAP", ascending=False)
       .reset_index(drop=True))
imp["순위"] = imp.index + 1
imp["기여도_%"] = (imp["mean_abs_SHAP"] / imp["mean_abs_SHAP"].sum() * 100).round(2)
imp.to_csv(TAB / "shap_importance.csv", index=False, encoding="utf-8-sig")
print(imp.head(15).to_string(index=False))

# ---------- (2) Beeswarm 요약도 ----------
plt.figure()
shap.summary_plot(sv, X_te, feature_names=feat_labels, max_display=15,
                  show=False, plot_size=(9, 7))
plt.title("SHAP 요약도 (beeswarm) — 기술료 발생 예측", fontsize=12)
savefig("shap_beeswarm.png")

# ---------- (3) 막대 중요도 ----------
plt.figure()
shap.summary_plot(sv, X_te, feature_names=feat_labels, plot_type="bar",
                  max_display=15, show=False, plot_size=(9, 7), color="#2f6f9f")
plt.title("SHAP 평균 절대 기여도 — 상위 15개 변수", fontsize=12)
savefig("shap_bar.png")

# ---------- (4) 의존성 그림 (상위 수치형 4개) ----------
num_top = [c for c in imp["변수"] if c in
           ("claim_total", "claim_independent", "ipc_count", "inventor_count",
            "family_count", "backward_citation", "backward_foreign", "특허연령_월",
            "등록연도")][:4]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, c in zip(axes.ravel(), num_top):
    j = list(X_te.columns).index(c)
    ax.scatter(X_te[c].values, sv[:, j], s=6, alpha=.25, color="#2f6f9f",
               edgecolors="none")
    ax.axhline(0, color="gray", lw=.8, ls="--")
    ax.set_xlabel(pretty(c)); ax.set_ylabel("SHAP 값")
    ax.set_title(pretty(c), fontsize=11)
    xq = np.quantile(X_te[c].values, 0.99)
    ax.set_xlim(X_te[c].min(), xq if xq > X_te[c].min() else None)
fig.suptitle("SHAP 의존성 그림 — 상위 수치형 변수 (상위1% 절단)", fontsize=13)
savefig("shap_dependence.png")

# ---------- 요약 메타 ----------
summary = {"model": "XGBoost", "test_N": int(len(X_te)),
           "base_value": float(np.asarray(explainer.expected_value).ravel()[0]),
           "top10": imp.head(10)[["라벨", "mean_abs_SHAP", "기여도_%"]]
           .to_dict("records")}
with open(MOD / "_shap_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("[s5] SHAP done.")
