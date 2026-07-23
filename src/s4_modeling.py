# -*- coding: utf-8 -*-
"""4단계: 모델 벤치마크 — LR·RF·XGBoost·LightGBM.
층화 train/test 분할 → 5-fold CV(train) → test 성능 → ROC/PR 곡선 → 모델 저장.
"""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, accuracy_score,
                             brier_score_loss, roc_curve, precision_recall_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from common import load_base, build_xy, SEED, TAB, FIG, MOD, savefig

# ---------- 데이터 ----------
df = load_base()
X, y = build_xy(df)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
spw = (y_tr == 0).sum() / (y_tr == 1).sum()   # scale_pos_weight
print(f"[s4] train={X_tr.shape} test={X_te.shape} pos_rate={y_tr.mean():.4f} spw={spw:.3f}")

# ---------- 모델 정의 ----------
models = {
    "LogisticRegression": Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=SEED)),
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    "XGBoost": XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="logloss", tree_method="hist", n_jobs=-1,
        random_state=SEED),
    "LightGBM": LGBMClassifier(
        n_estimators=600, max_depth=-1, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        n_jobs=-1, random_state=SEED, verbose=-1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ---------- 학습 · CV · 평가 ----------
rows, fitted, curves = [], {}, {}
for name, model in models.items():
    print(f"  ▶ {name} …", flush=True)
    cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    rows.append({
        "모델": name,
        "CV_ROC_AUC_평균": round(cv_auc.mean(), 4),
        "CV_ROC_AUC_표준편차": round(cv_auc.std(), 4),
        "Test_ROC_AUC": round(roc_auc_score(y_te, p), 4),
        "Test_PR_AUC": round(average_precision_score(y_te, p), 4),
        "Test_Accuracy": round(accuracy_score(y_te, pred), 4),
        "Test_Precision": round(precision_score(y_te, pred), 4),
        "Test_Recall": round(recall_score(y_te, pred), 4),
        "Test_F1": round(f1_score(y_te, pred), 4),
        "Test_Brier": round(brier_score_loss(y_te, p), 4),
    })
    fitted[name] = model
    curves[name] = p
    with open(MOD / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

perf = pd.DataFrame(rows).sort_values("Test_ROC_AUC", ascending=False)
perf.to_csv(TAB / "model_performance.csv", index=False, encoding="utf-8-sig")
print(perf.to_string(index=False))

# ---------- 최적 모델 기록 (SHAP 단계용) ----------
best = perf.iloc[0]["모델"]
meta = {"best_model": best, "scale_pos_weight": float(spw),
        "n_features": int(X.shape[1]), "test_size": 0.2, "seed": SEED,
        "features": list(X.columns)}
with open(MOD / "_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
# 재현용 test 분할 인덱스 저장
np.save(MOD / "test_index.npy", X_te.index.values)
print(f"[s4] best model = {best}")

# ---------- ROC / PR 곡선 ----------
base_rate = y_te.mean()
colors = {"LogisticRegression": "#9aa7b5", "RandomForest": "#6b9b37",
          "XGBoost": "#2f6f9f", "LightGBM": "#c8792f"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
for name, p in curves.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    auc = roc_auc_score(y_te, p)
    ax1.plot(fpr, tpr, lw=1.8, color=colors[name], label=f"{name} (AUC={auc:.3f})")
    prec, rec, _ = precision_recall_curve(y_te, p)
    ap = average_precision_score(y_te, p)
    ax2.plot(rec, prec, lw=1.8, color=colors[name], label=f"{name} (AP={ap:.3f})")
ax1.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax1.set_xlabel("거짓 양성률 (FPR)"); ax1.set_ylabel("참 양성률 (TPR)")
ax1.set_title("ROC 곡선", fontsize=12); ax1.legend(fontsize=9, loc="lower right")
ax2.axhline(base_rate, ls="--", color="gray", lw=1,
            label=f"무작위 기준 {base_rate:.3f}")
ax2.set_xlabel("재현율 (Recall)"); ax2.set_ylabel("정밀도 (Precision)")
ax2.set_title("Precision–Recall 곡선", fontsize=12); ax2.legend(fontsize=9, loc="upper right")
fig.suptitle("모델 성능 비교 (Test set, N={:,})".format(len(y_te)), fontsize=13)
savefig("roc_pr.png")
print("[s4] modeling done.")
