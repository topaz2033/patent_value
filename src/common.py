# -*- coding: utf-8 -*-
"""공통 설정: 경로, 한글 폰트, 피처 정의, 데이터 로드/전처리."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
TAB = OUT / "tables"
FIG = OUT / "figures"
MOD = OUT / "models"
PAPER = ROOT / "paper"
for d in (TAB, FIG, MOD):
    d.mkdir(parents=True, exist_ok=True)

# 한글 폰트 (Windows: Malgun Gothic)
for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        matplotlib.rcParams["font.family"] = f
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["savefig.bbox"] = "tight"

SEED = 42

TARGET = "기술료발생"

NUM_FEATURES = [
    "claim_total", "claim_independent", "ipc_count", "inventor_count",
    "family_count", "backward_citation", "backward_foreign", "특허연령_월",
]
CAT_FEATURES = ["기관유형", "부처유형", "finalDisposal", "registerStatus"]
YEAR_FEATURES = ["등록연도"]  # 출원연도와 강상관 → 등록연도만 사용
FLAG_FEATURES = ["출원일자_결측"]

# 사람이 읽는 라벨 (그림/표용)
LABELS = {
    "claim_total": "청구항 수",
    "claim_independent": "독립 청구항 수",
    "ipc_count": "IPC 분류 수",
    "inventor_count": "발명자 수",
    "family_count": "패밀리 수",
    "backward_citation": "후방 인용 수",
    "backward_foreign": "해외 후방 인용 수",
    "특허연령_월": "특허 연령(개월)",
    "기관유형": "기관유형",
    "부처유형": "부처유형",
    "finalDisposal": "최종처분",
    "registerStatus": "등록상태",
    "등록연도": "등록연도",
    "출원일자_결측": "출원일자 결측",
}


def load_base():
    """모델링 베이스 로드 + 최소 전처리."""
    df = pd.read_parquet(DATA / "analysis_base_v2.parquet")
    # 특허연령/출원연도 결측 → 중앙값 대체 (결측 플래그는 출원일자_결측이 이미 존재)
    df["특허연령_월"] = df["특허연령_월"].fillna(df["특허연령_월"].median())
    df["출원연도"] = df["출원연도"].fillna(df["출원연도"].median())
    # 범주형 문자열 정리
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)
    return df


def build_xy(df, encode="onehot"):
    """모델링용 X, y 구성.
    encode='onehot' → LR/RF/XGB 공통 수치 행렬 (get_dummies)
    """
    feats_num = NUM_FEATURES + FLAG_FEATURES + YEAR_FEATURES
    X_num = df[feats_num].astype(float)
    X_cat = pd.get_dummies(df[CAT_FEATURES], prefix=CAT_FEATURES, drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    X.columns = [str(c) for c in X.columns]
    y = df[TARGET].astype(int).values
    return X, y


def savefig(name):
    p = FIG / name
    plt.savefig(p)
    plt.close()
    return p
