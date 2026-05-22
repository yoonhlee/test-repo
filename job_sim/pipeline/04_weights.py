"""
04_weights.py
activities / abilities / personalities 3종 데이터를 결합해 Z 표준화 후,
시뮬레이션 대상 5개 군집별로 5축 가중치를 계산한다.
Softmax(temperature=0.5) 로 정규화.

産出物:
  data/processed/cluster_weights.json
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- パス設定 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

# 군집 이름 매핑 (클러스터링 결과 기반)
CLUSTER_NAMES = {
    "C1": "리더십·인사관리",
    "C2": "데이터·IT",
    "C4": "정보·행정서비스",
    "C5": "기술·현장분석",
    "C6": "대인서비스",
}

# 시뮬레이션 대상 군집 ID
SIM_CLUSTERS = list(CLUSTER_NAMES.keys())


def softmax(x: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """温度付き Softmax"""
    x = np.array(x, dtype=float)
    x = x / temperature
    x = x - x.max()  # 数値安定化
    e = np.exp(x)
    return e / e.sum()


def main():
    # ── 데이터 로드 ────────────────────────────────────────────────
    df_act = pd.read_csv(
        os.path.join(PROC_DIR, "activities.csv"),
        index_col="job_code", encoding="utf-8-sig"
    )
    df_abl = pd.read_csv(
        os.path.join(PROC_DIR, "abilities.csv"),
        index_col="job_code", encoding="utf-8-sig"
    )
    df_per = pd.read_csv(
        os.path.join(PROC_DIR, "personalities.csv"),
        index_col="job_code", encoding="utf-8-sig"
    )
    df_clus = pd.read_csv(
        os.path.join(PROC_DIR, "clusters.csv"),
        encoding="utf-8-sig"
    )
    with open(os.path.join(PROC_DIR, "axis_mapping.json"), "r", encoding="utf-8") as f:
        axis_mapping = json.load(f)

    # job_name 열 제거 후 특징 행렬 결합
    df_act_feat = df_act.drop(columns=["job_name"])
    df_abl_feat = df_abl.drop(columns=["job_name"])
    df_per_feat = df_per.drop(columns=["job_name"])

    # 공통 직업코드 교집합
    common_idx = df_act_feat.index.intersection(df_abl_feat.index).intersection(df_per_feat.index)
    df_combined = pd.concat(
        [df_act_feat.loc[common_idx], df_abl_feat.loc[common_idx], df_per_feat.loc[common_idx]],
        axis=1
    )
    print(f"결합 행렬: {df_combined.shape[0]} 직업 × {df_combined.shape[1]} 항목")

    # ── Z 표준화 ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_z = pd.DataFrame(
        scaler.fit_transform(df_combined.values),
        index=df_combined.index,
        columns=df_combined.columns
    )

    # ── 축별 활동 항목 목록 (activities.csv 의 항목만 사용) ──────
    act_cols = set(df_act_feat.columns)
    ax_items = {}
    for ax_id, ax_def in axis_mapping.items():
        valid = [it for it in ax_def["items"] if it in act_cols]
        ax_items[ax_id] = valid
        print(f"  {ax_id} ({ax_def['label']}): {len(valid)}개 항목")

    # ── 군집별 Z-score 평균 → 가중치 계산 ───────────────────────
    clus_dict = df_clus.set_index("job_code")["cluster_id"].to_dict()
    X_z["cluster_id"] = X_z.index.map(clus_dict)

    cluster_weights = {}

    for cid in SIM_CLUSTERS:
        mask   = X_z["cluster_id"] == cid
        subset = X_z[mask].drop(columns=["cluster_id"])
        n_jobs = mask.sum()
        print(f"\n군집 {cid} ({CLUSTER_NAMES[cid]}): {n_jobs}개 직업")

        # 각 축의 항목 평균 Z-score → 축 대표값
        ax_scores = {}
        for ax_id in ["AX1", "AX2", "AX3", "AX4", "AX5"]:
            items = ax_items.get(ax_id, [])
            if not items:
                ax_scores[ax_id] = 0.0
                continue
            valid_items = [it for it in items if it in subset.columns]
            if not valid_items:
                ax_scores[ax_id] = 0.0
                continue
            ax_scores[ax_id] = float(subset[valid_items].values.mean())

        print(f"  원시 Z-score: { {k: round(v,3) for k, v in ax_scores.items()} }")

        # Softmax 정규화
        raw_vals  = np.array([ax_scores[ax] for ax in ["AX1","AX2","AX3","AX4","AX5"]])
        norm_vals = softmax(raw_vals, temperature=0.5)
        weights   = {f"AX{i+1}": round(float(norm_vals[i]), 4) for i in range(5)}

        print(f"  Softmax 가중치: {weights}  (합={sum(weights.values()):.4f})")

        cluster_weights[cid] = {
            "name":  CLUSTER_NAMES[cid],
            "n_jobs": int(n_jobs),
            **weights,
        }

    # ── JSON 저장 ─────────────────────────────────────────────────
    out_path = os.path.join(PROC_DIR, "cluster_weights.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cluster_weights, f, ensure_ascii=False, indent=2)

    print(f"\n[04] cluster_weights.json 저장 → {out_path}")
    print("\n[ 최종 군집별 가중치 ]")
    for cid, val in cluster_weights.items():
        print(f"  {cid} {val['name']}: "
              f"AX1={val['AX1']:.3f} AX2={val['AX2']:.3f} AX3={val['AX3']:.3f} "
              f"AX4={val['AX4']:.3f} AX5={val['AX5']:.3f}")


if __name__ == "__main__":
    main()
