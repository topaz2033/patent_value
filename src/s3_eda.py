# -*- coding: utf-8 -*-
"""3단계: EDA — 기초통계, 그룹 비교(t-검정/χ²), 상관관계, 그림."""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from common import (load_base, NUM_FEATURES, CAT_FEATURES, TARGET, TAB, FIG,
                    LABELS, savefig)

df = load_base()
y = df[TARGET]

# ---------- (1) 타깃 분포 ----------
vc = df[TARGET].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["기술료 미발생 (0)", "기술료 발생 (1)"], vc.values,
              color=["#9aa7b5", "#2f6f9f"])
for b, v in zip(bars, vc.values):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:,}\n({v/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=10)
ax.set_title("타깃 분포: 기술료 발생 여부", fontsize=12)
ax.set_ylabel("특허 수")
ax.set_ylim(0, vc.max()*1.15)
savefig("eda_target_dist.png")

# ---------- (2) 수치형: 발생 vs 비발생 그룹 비교 (t-검정 + Mann-Whitney) ----------
rows = []
for c in NUM_FEATURES:
    a = df.loc[y == 1, c].dropna()
    b = df.loc[y == 0, c].dropna()
    t, pt = stats.ttest_ind(a, b, equal_var=False)
    u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Cohen's d
    ns = a.std()**2*(len(a)-1) + b.std()**2*(len(b)-1)
    sp = np.sqrt(ns/(len(a)+len(b)-2))
    d = (a.mean()-b.mean())/sp if sp > 0 else 0
    rows.append({
        "변수": c, "라벨": LABELS.get(c, c),
        "발생_평균": round(a.mean(), 3), "비발생_평균": round(b.mean(), 3),
        "차이": round(a.mean()-b.mean(), 3), "Cohens_d": round(d, 4),
        "t": round(t, 2), "p_ttest": f"{pt:.2e}", "p_mwu": f"{pu:.2e}",
    })
grp_num = pd.DataFrame(rows)
grp_num.to_csv(TAB / "eda_group_compare_numeric.csv", index=False, encoding="utf-8-sig")

# ---------- (3) 범주형: χ² 검정 + 타깃율 ----------
rows = []
for c in CAT_FEATURES + ["부처유형"]:
    if c not in df.columns:
        continue
    ct = pd.crosstab(df[c], y)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape)-1)))
    rows.append({"변수": c, "라벨": LABELS.get(c, c), "chi2": round(chi2, 1),
                 "dof": dof, "p": f"{p:.2e}", "CramersV": round(cramers_v, 4)})
grp_cat = pd.DataFrame(rows).drop_duplicates("변수")
grp_cat.to_csv(TAB / "eda_group_compare_categorical.csv", index=False, encoding="utf-8-sig")

# 기관유형별 타깃율
for c in ["기관유형", "부처유형", "등록연도", "registerStatus"]:
    g = df.groupby(c)[TARGET].agg(발생률="mean", 건수="count").reset_index()
    g["발생률"] = (g["발생률"]*100).round(2)
    g = g.sort_values("발생률", ascending=False)
    g.to_csv(TAB / f"eda_rate_by_{c}.csv", index=False, encoding="utf-8-sig")

# ---------- (4) 상관관계 히트맵 ----------
corr = df[NUM_FEATURES + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(8, 6.5))
labs = [LABELS.get(c, c) for c in NUM_FEATURES] + ["기술료발생"]
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            xticklabels=labs, yticklabels=labs, ax=ax,
            cbar_kws={"shrink": .8}, annot_kws={"size": 8})
ax.set_title("수치형 변수 상관행렬 (Pearson)", fontsize=12)
plt.xticks(rotation=45, ha="right")
savefig("eda_corr_heatmap.png")

# ---------- (5) 기관유형/부처유형 타깃율 막대 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, c, ttl in zip(axes, ["기관유형", "부처유형"], ["기관유형별", "부처유형별"]):
    g = df.groupby(c)[TARGET].mean().sort_values(ascending=False)*100
    ax.bar(range(len(g)), g.values, color="#2f6f9f")
    ax.axhline(y.mean()*100, color="crimson", ls="--", lw=1,
               label=f"전체 평균 {y.mean()*100:.1f}%")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("기술료 발생률(%)")
    ax.set_title(f"{ttl} 기술료 발생률", fontsize=12)
    ax.legend(fontsize=9)
savefig("eda_rate_by_group.png")

# ---------- (6) 등록연도별 발생률 추이 ----------
g = df.groupby("등록연도")[TARGET].agg(["mean", "count"])
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.bar(g.index, g["count"], color="#cfd8e3", label="특허 수")
ax1.set_ylabel("특허 수", color="#5a6b7b")
ax2 = ax1.twinx()
ax2.plot(g.index, g["mean"]*100, "o-", color="#2f6f9f", lw=2, label="발생률")
ax2.set_ylabel("기술료 발생률(%)", color="#2f6f9f")
ax1.set_xlabel("등록연도")
ax1.set_title("등록연도별 특허 수와 기술료 발생률", fontsize=12)
savefig("eda_rate_by_year.png")

# ---------- (7) 주요 수치형 분포 (발생/비발생 겹침) ----------
show = ["claim_total", "ipc_count", "inventor_count", "backward_citation"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, c in zip(axes.ravel(), show):
    hi = df.loc[y == 1, c]
    lo = df.loc[y == 0, c]
    q = df[c].quantile(0.99)
    bins = np.linspace(df[c].min(), q, 30)
    ax.hist(lo, bins=bins, alpha=.5, density=True, label="미발생", color="#9aa7b5")
    ax.hist(hi, bins=bins, alpha=.5, density=True, label="발생", color="#2f6f9f")
    ax.set_title(LABELS.get(c, c), fontsize=11)
    ax.legend(fontsize=9)
fig.suptitle("주요 수치형 변수 분포 (발생 vs 미발생, 상위1% 절단)", fontsize=12)
savefig("eda_num_dist.png")

print("[s3] EDA done.")
print(grp_num.to_string(index=False))
print(grp_cat.to_string(index=False))
