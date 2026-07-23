# -*- coding: utf-8 -*-
"""2단계: 변수 정의 기술통계 표 산출."""
import pandas as pd
from common import load_base, NUM_FEATURES, CAT_FEATURES, YEAR_FEATURES, TARGET, TAB, LABELS

df = load_base()

# 수치형 요약
rows = []
for c in NUM_FEATURES:
    s = df[c]
    rows.append({
        "변수": c, "라벨": LABELS.get(c, c), "유형": "수치",
        "평균": round(s.mean(), 3), "표준편차": round(s.std(), 3),
        "최소": round(s.min(), 1), "중앙값": round(s.median(), 1),
        "최대": round(s.max(), 1), "결측": int(s.isna().sum()),
    })
num_tab = pd.DataFrame(rows)
num_tab.to_csv(TAB / "variable_summary_numeric.csv", index=False, encoding="utf-8-sig")

# 범주형 요약 (범주 수 + 타깃율 범위)
rows = []
for c in CAT_FEATURES + YEAR_FEATURES:
    g = df.groupby(c)[TARGET].agg(["mean", "count"])
    rows.append({
        "변수": c, "라벨": LABELS.get(c, c), "유형": "범주",
        "범주수": df[c].nunique(),
        "타깃율_최소": round(g["mean"].min(), 4),
        "타깃율_최대": round(g["mean"].max(), 4),
        "최빈범주": df[c].value_counts().idxmax(),
    })
cat_tab = pd.DataFrame(rows)
cat_tab.to_csv(TAB / "variable_summary_categorical.csv", index=False, encoding="utf-8-sig")

print("[s2] numeric vars:", len(num_tab), "| categorical vars:", len(cat_tab))
print(num_tab.to_string(index=False))
print(cat_tab.to_string(index=False))
