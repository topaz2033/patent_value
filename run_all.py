# run_all.py
# 전체 파이프라인 자동 실행 (src/s2~s6 순차)
# 사용: venv 활성화 후  ->  python run_all.py

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
scripts = [
    "s2_variables.py",   # 2단계: 변수 정의 기술통계
    "s3_eda.py",         # 3단계: EDA
    "s4_modeling.py",    # 4단계: 모델 벤치마크
    "s5_shap.py",        # 5단계: SHAP 해석
    "s6_robustness.py",  # 6단계: 강건성
]

print("=" * 50)
print("Patent Value 분석 파이프라인 시작")
print("=" * 50)

for s in scripts:
    p = SRC / s
    if p.exists():
        print(f"\n▶ 실행 중: src/{s}")
        # common.py를 모듈로 임포트하므로 cwd=src 에서 실행
        result = subprocess.run([sys.executable, s], cwd=SRC,
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ 완료: src/{s}")
        else:
            print(f"  ❌ 오류: src/{s}")
            print(result.stderr)
            sys.exit(1)
    else:
        print(f"  ⏭ 건너뜀 (미작성): src/{s}")

print("\n" + "=" * 50)
print("✅ 전체 파이프라인 완료 → output/tables, output/figures, output/models")
print("=" * 50)
