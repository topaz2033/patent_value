# -*- coding: utf-8 -*-
"""7단계(민감도): 등록연도 제외 재분석.
관측창(right-censoring) 우려 대응 — 등록연도를 제거하고 4개 모델을 동일 분할로
재학습해 AUC 변화를 비교. 등록연도 없는 XGBoost의 SHAP 상위도 재산출.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from common import load_base, build_xy, LABELS, SEED, TAB, FIG, MOD, savefig

# ---------- 데이터: 등록연도 제외 ----------
df = load_base()
X, y = build_xy(df)
assert "등록연도" in X.columns
X_ny = X.drop(columns=["등록연도"])
X_tr, X_te, y_tr, y_te = train_test_split(
    X_ny, y, test_size=0.2, stratify=y, random_state=SEED)   # 원본과 동일 분할(행 동일)
spw = (y_tr == 0).sum() / (y_tr == 1).sum()
print(f"[s7] no-year X={X_ny.shape} (등록연도 제거), spw={spw:.3f}")

def make_models():
    return {
        "LogisticRegression": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=SEED))]),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMClassifier(
            n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=-1,
            random_state=SEED, verbose=-1),
    }

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ---------- 원본 성능 로드 ----------
orig = pd.read_csv(TAB / "model_performance.csv").set_index("모델")

# ---------- 등록연도 제외 학습·평가 ----------
rows = {}
xgb_ny = None
for name, model in make_models().items():
    print(f"  ▶ {name} (no-year) …", flush=True)
    cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    rows[name] = {"CV_AUC_ny": cv_auc.mean(),
                  "Test_AUC_ny": roc_auc_score(y_te, p),
                  "Test_PRAUC_ny": average_precision_score(y_te, p)}
    if name == "XGBoost":
        xgb_ny = model

# ---------- 비교표 ----------
comp = []
for name in ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]:
    a0 = orig.loc[name, "Test_ROC_AUC"]
    a1 = rows[name]["Test_AUC_ny"]
    p0 = orig.loc[name, "Test_PR_AUC"]
    p1 = rows[name]["Test_PRAUC_ny"]
    comp.append({
        "모델": name,
        "AUC_원본": round(a0, 4), "AUC_등록연도제외": round(a1, 4),
        "ΔAUC": round(a1 - a0, 4),
        "PRAUC_원본": round(p0, 4), "PRAUC_등록연도제외": round(p1, 4),
        "ΔPRAUC": round(p1 - p0, 4),
    })
comp = pd.DataFrame(comp)
comp.to_csv(TAB / "sensitivity_registration_year.csv", index=False, encoding="utf-8-sig")
print(comp.to_string(index=False))

# ---------- 등록연도 제외 XGBoost의 SHAP ----------
def pretty(col):
    if col in LABELS:
        return LABELS[col]
    for base in ("기관유형_", "부처유형_", "finalDisposal_", "registerStatus_"):
        if col.startswith(base):
            return f"{LABELS.get(base[:-1], base[:-1])}={col[len(base):]}"
    return col

expl = shap.TreeExplainer(xgb_ny)
sv = expl.shap_values(X_te)
mean_abs = np.abs(sv).mean(axis=0)
imp = (pd.DataFrame({"변수": X_te.columns, "라벨": [pretty(c) for c in X_te.columns],
                     "mean_abs_SHAP": mean_abs})
       .sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True))
imp["순위"] = imp.index + 1
imp["기여도_%"] = (imp["mean_abs_SHAP"] / imp["mean_abs_SHAP"].sum() * 100).round(2)
imp.to_csv(TAB / "shap_importance_noyear.csv", index=False, encoding="utf-8-sig")
print("\n[등록연도 제외 SHAP 상위 10]")
print(imp.head(10)[["순위", "라벨", "기여도_%"]].to_string(index=False))

# ---------- 그림: 원본 vs 등록연도 제외 AUC ----------
fig, ax = plt.subplots(figsize=(8, 5))
names = comp["모델"].tolist()
x = np.arange(len(names)); w = 0.38
ax.bar(x - w/2, comp["AUC_원본"], w, label="원본(등록연도 포함)", color="#2f6f9f")
ax.bar(x + w/2, comp["AUC_등록연도제외"], w, label="등록연도 제외", color="#c8792f")
for i, (a0, a1) in enumerate(zip(comp["AUC_원본"], comp["AUC_등록연도제외"])):
    ax.text(i - w/2, a0, f"{a0:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + w/2, a1, f"{a1:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
ax.set_ylabel("Test ROC-AUC"); ax.set_ylim(0.5, 0.75)
ax.set_title("민감도 분석: 등록연도 제외 시 성능 변화", fontsize=12)
ax.legend(fontsize=9)
savefig("sensitivity_registration_year.png")
print("\n[s7] sensitivity done.")
