# -*- coding: utf-8 -*-
"""6단계: 강건성 검증.
(A) 하위집단 성능: 등록연도별·기관유형별 test AUC/PR-AUC (주 모델 XGBoost).
(B) 시계열 외삽(out-of-time): 과거연도 학습 → 미래연도 예측으로 시간 일반화 확인.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from common import load_base, build_xy, SEED, TAB, FIG, savefig

df = load_base()
X, y = build_xy(df)
y = pd.Series(y, index=X.index)

# ---------- (A) 하위집단 성능 (기존 학습 모델 재사용) ----------
with open("../output/models/XGBoost.pkl", "rb") as f:
    model = pickle.load(f)
test_idx = np.load("../output/models/test_index.npy")
X_te, y_te = X.loc[test_idx], y.loc[test_idx]
p_te = model.predict_proba(X_te)[:, 1]

overall_auc = roc_auc_score(y_te, p_te)
overall_ap = average_precision_score(y_te, p_te)
print(f"[s6] overall test AUC={overall_auc:.4f} PR-AUC={overall_ap:.4f}")

def subgroup_table(group_col):
    g = df.loc[test_idx, group_col]
    rows = []
    for key in sorted(g.unique(), key=lambda v: str(v)):
        m = (g == key).values
        yy, pp = y_te.values[m], p_te[m]
        if len(np.unique(yy)) < 2 or len(yy) < 50:
            continue
        rows.append({group_col: key, "N": int(m.sum()),
                     "발생률": round(float(yy.mean()), 4),
                     "ROC_AUC": round(roc_auc_score(yy, pp), 4),
                     "PR_AUC": round(average_precision_score(yy, pp), 4)})
    return pd.DataFrame(rows)

sub_year = subgroup_table("등록연도")
sub_year.to_csv(TAB / "robustness_by_year.csv", index=False, encoding="utf-8-sig")
sub_inst = subgroup_table("기관유형")
sub_inst.to_csv(TAB / "robustness_by_institution.csv", index=False, encoding="utf-8-sig")
print(sub_year.to_string(index=False))
print(sub_inst.to_string(index=False))

# ---------- (B) 시계열 외삽 (out-of-time) ----------
spw = (y == 0).sum() / (y == 1).sum()
cutoffs = [2019, 2020, 2021]
oot_rows = []
for cut in cutoffs:
    tr = df["등록연도"] <= cut
    te = df["등록연도"] > cut
    if te.sum() < 100:
        continue
    m = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                      eval_metric="logloss", tree_method="hist", n_jobs=-1,
                      random_state=SEED)
    m.fit(X[tr.values], y[tr.values])
    pp = m.predict_proba(X[te.values])[:, 1]
    oot_rows.append({
        "학습": f"≤{cut}", "예측": f"{cut+1}~2023",
        "학습N": int(tr.sum()), "예측N": int(te.sum()),
        "ROC_AUC": round(roc_auc_score(y[te.values], pp), 4),
        "PR_AUC": round(average_precision_score(y[te.values], pp), 4)})
oot = pd.DataFrame(oot_rows)
oot.to_csv(TAB / "robustness_out_of_time.csv", index=False, encoding="utf-8-sig")
print(oot.to_string(index=False))

# ---------- 그림: 하위집단 AUC ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.bar(sub_year["등록연도"].astype(str), sub_year["ROC_AUC"], color="#2f6f9f")
ax1.axhline(overall_auc, color="crimson", ls="--", lw=1,
            label=f"전체 {overall_auc:.3f}")
ax1.set_ylim(0.5, max(0.8, sub_year["ROC_AUC"].max()+0.03))
ax1.set_ylabel("ROC-AUC"); ax1.set_xlabel("등록연도")
ax1.set_title("등록연도별 성능", fontsize=12); ax1.legend(fontsize=9)
ax2.bar(sub_inst["기관유형"].astype(str), sub_inst["ROC_AUC"], color="#6b9b37")
ax2.axhline(overall_auc, color="crimson", ls="--", lw=1,
            label=f"전체 {overall_auc:.3f}")
ax2.set_ylim(0.5, max(0.8, sub_inst["ROC_AUC"].max()+0.03))
ax2.set_ylabel("ROC-AUC"); ax2.set_xlabel("기관유형")
ax2.set_title("기관유형별 성능", fontsize=12); ax2.legend(fontsize=9)
fig.suptitle("하위집단 강건성 (XGBoost, test set)", fontsize=13)
savefig("robustness_subgroup.png")

# ---------- 그림: 시계열 외삽 ----------
if len(oot):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(oot))
    ax.plot(x, oot["ROC_AUC"], "o-", color="#2f6f9f", lw=2, label="ROC-AUC")
    ax.plot(x, oot["PR_AUC"], "s--", color="#c8792f", lw=2, label="PR-AUC")
    ax.axhline(overall_auc, color="crimson", ls=":", lw=1,
               label=f"무작위분할 AUC {overall_auc:.3f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r['학습']}→{r['예측']}" for _, r in oot.iterrows()],
                       fontsize=9)
    ax.set_ylabel("성능"); ax.set_ylim(0.4, 0.8)
    ax.set_title("시계열 외삽 검증 (과거 학습 → 미래 예측)", fontsize=12)
    ax.legend(fontsize=9)
    savefig("robustness_out_of_time.png")

print("[s6] robustness done.")
