"""
03_factor_analysis.py
activities.csv の 41 項目に対して要因分析を実施し、
5 つの思考軸 (AX1~AX5) へ KNOW 活動項目をマッピングする。

産出物:
  data/processed/axis_mapping.json
  {
    "AX1": {"label": "정보분석·논리", "items": [...]},
    ...
  }
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from factor_analyzer import FactorAnalyzer
    # sklearn 1.8+ 호환성 체크 (force_all_finite 인자 제거됨)
    import inspect
    from sklearn.utils.validation import check_array
    if "force_all_finite" not in inspect.signature(check_array).parameters:
        # monkey-patch: factor_analyzer 내부의 check_array 호출을 우회
        import factor_analyzer.factor_analyzer as _fa_mod
        _orig_check = _fa_mod.check_array
        def _compat_check_array(X, **kwargs):
            kwargs.pop("force_all_finite", None)
            return _orig_check(X, **kwargs)
        _fa_mod.check_array = _compat_check_array
    HAS_FA = True
except (ImportError, Exception) as e:
    HAS_FA = False
    print(f"[경고] factor_analyzer 미사용 ({e}) → 하드코딩 매핑만 사용")

# --- パス設定 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

# ─────────────────────────────────────────────────────────────
# 5 사고방식 축 정의 (KNOW 활동 항목명과 유사한 것으로 매핑)
# 실제 XML 에서 확인된 41개 활동명을 기준으로 작성
# ─────────────────────────────────────────────────────────────
AXIS_DEFINITION = {
    "AX1": {
        "label": "정보분석·논리",
        "keywords": [
            "정보 수집",
            "정보, 자료 분석",
            "정보 처리",
            "컴퓨터 업무",
            "기준에 따른 정보 평가",
            "정보의 의미 해석",
            "정보 작성, 기록",
        ],
    },
    "AX2": {
        "label": "관찰·탐색",
        "keywords": [
            "절차, 자료, 주변환경 관찰",
            "사물, 행동, 사건 파악",
            "새로운 지식의 습득, 활용",
            "장비, 건축물, 자재 검사",
            "제품, 사건, 정보의 수치 추정",
        ],
    },
    "AX3": {
        "label": "전략·판단",
        "keywords": [
            "의사 결정, 문제점 해결",
            "목표, 전략 수립",
            "업무 계획, 우선순위 결정",
            "업무, 활동에 대한 일정관리",
            "창조적 생각",
        ],
    },
    "AX4": {
        "label": "리더십·조직",
        "keywords": [
            "부하 직원들에게 업무 안내, 지시, 동기부여",
            "팀 구성, 협업 촉진",
            "사람들의 업무와 활동을 조직, 편성",
            "인사 업무",
            "자원 관리",
            "행정, 관리 업무",
            "사람들의 능력 개발, 지도",
        ],
    },
    "AX5": {
        "label": "대인서비스",
        "keywords": [
            "대인관계 유지",
            "업무상 사람들을 직접 응대",
            "사람들을 배려, 돌봄",
            "상사, 동료, 부하직원과 소통",
            "사람들에게 조언, 상담",
            "협상, 갈등 해결",
            "사람들에게 영향력 행사",
        ],
    },
}


def fuzzy_match(col_name: str, keywords: list[str]) -> bool:
    """列名がキーワードリストのいずれかと部分一致するか判定"""
    c = col_name.strip()
    for kw in keywords:
        if kw in c or c in kw:
            return True
    return False


def main():
    csv_path = os.path.join(PROC_DIR, "activities.csv")
    df = pd.read_csv(csv_path, index_col="job_code", encoding="utf-8-sig")
    df_feat = df.drop(columns=["job_name"])
    all_cols = df_feat.columns.tolist()

    print(f"活動項目数: {len(all_cols)}")

    # ─── 要因分析 (factor_analyzer がある場合) ─────────────────
    if HAS_FA:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_feat.values)

        # 1次: Kaiser Rule (고유값 > 1)
        fa1 = FactorAnalyzer(n_factors=len(all_cols), rotation=None)
        fa1.fit(X)
        ev, _ = fa1.get_eigenvalues()
        n_factors_kaiser = int((ev > 1).sum())
        print(f"Kaiser Rule 요인 수: {n_factors_kaiser}")

        # 2차: 요인수 제한 분석 (회전: varimax)
        n_f = max(5, min(n_factors_kaiser, 10))
        fa2 = FactorAnalyzer(n_factors=n_f, rotation="varimax")
        fa2.fit(X)
        loadings = pd.DataFrame(
            fa2.loadings_,
            index=all_cols,
            columns=[f"F{i+1}" for i in range(n_f)],
        )

        # F1 に대한 부하량 |0.4| 이상 항목 표시
        f1_high = loadings[loadings["F1"].abs() >= 0.4].index.tolist()
        print(f"\nF1 고부하량 항목 ({len(f1_high)}개): {f1_high}")
    else:
        print("요인분석 생략 — 하드코딩 매핑만 사용합니다.")

    # ─── 軸マッピング ─────────────────────────────────────────
    axis_mapping = {}
    used_cols = set()

    for ax_id, ax_def in AXIS_DEFINITION.items():
        matched = [
            col for col in all_cols
            if fuzzy_match(col, ax_def["keywords"]) and col not in used_cols
        ]
        used_cols.update(matched)
        axis_mapping[ax_id] = {
            "label": ax_def["label"],
            "items": matched,
        }
        print(f"\n{ax_id} ({ax_def['label']})  매핑 항목 {len(matched)}개:")
        for item in matched:
            print(f"    · {item}")

    # 미매핑 항목 확인
    unmapped = [c for c in all_cols if c not in used_cols]
    if unmapped:
        print(f"\n[미매핑 항목 {len(unmapped)}개]:")
        for item in unmapped:
            print(f"    (미사용) {item}")

    # JSON 저장
    out_path = os.path.join(PROC_DIR, "axis_mapping.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(axis_mapping, f, ensure_ascii=False, indent=2)

    print(f"\n[03] axis_mapping.json 저장 → {out_path}")


if __name__ == "__main__":
    main()
