"""
02_clustering.py
activities.csv (41종) を標準化して k-means (k=8) でクラスタリング。
Hopkins 統計量と Silhouette score を出力。
제외 키워드("신체활동", "물건조종", "기계제어") に該当する군집を除外し、
残り 5 군집를 シミュレーション対象として選定。

産出物:
  data/processed/clusters.csv
    columns: job_code, job_name, cluster_id, top_activities, is_simulation_target
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- パス設定 ---
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")

RANDOM_SEED = 42
K = 8


# ── Hopkins 統計量 ──────────────────────────────────────────────
def hopkins_statistic(X: np.ndarray, sample_ratio: float = 0.1) -> float:
    """
    Hopkins 統計量を計算する (クラスタリング傾向の検定)。
    H が 0.5 に近いほどランダム、1 に近いほどクラスタ構造が強い。
    """
    n, d = X.shape
    m = max(1, int(n * sample_ratio))
    rng = random.Random(RANDOM_SEED)

    # ランダムサンプルを選ぶ
    idx = rng.sample(range(n), m)
    X_s = X[idx]

    # データ範囲内でランダム点を生成
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    X_r  = np.column_stack([
        rng.uniform(lo, hi) for lo, hi in zip(mins, maxs)
        for _ in range(1)
    ])  # ← scalar loop workaround
    # 正しい形状で生成
    X_r = np.array([
        [rng.uniform(mins[j], maxs[j]) for j in range(d)]
        for _ in range(m)
    ])

    # 各点の最近傍距離 (ユークリッド)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(X)

    # u: X_r → データ点への最近傍距離
    d_r, _ = nn.kneighbors(X_r)
    u = d_r[:, 0]

    # w: X_s → 他のデータ点への最近傍距離 (自身を除く)
    d_s, _ = nn.kneighbors(X_s)
    w = d_s[:, 1]

    H = u.sum() / (u.sum() + w.sum() + 1e-12)
    return float(H)


def main():
    # --- データ読み込み ---
    csv_path = os.path.join(PROC_DIR, "activities.csv")
    df = pd.read_csv(csv_path, index_col="job_code", encoding="utf-8-sig")

    job_names = df["job_name"].copy()
    df_feat   = df.drop(columns=["job_name"])
    feature_cols = df_feat.columns.tolist()

    # --- 標準化 ---
    scaler = StandardScaler()
    X = scaler.fit_transform(df_feat.values)

    # --- Hopkins 統計量 ---
    H = hopkins_statistic(X)
    print(f"Hopkins 統計量: {H:.4f}  (>0.75 = クラスタ傾向あり)")

    # --- k-means (k=8) ---
    km = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=20, max_iter=500)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)
    print(f"Silhouette Score (k={K}): {sil:.4f}")

    # --- 군집별 상위 5 활동 추출 ---
    cluster_top = {}
    for cid in range(K):
        mask   = labels == cid
        center = X[mask].mean(axis=0)
        top5   = sorted(zip(feature_cols, center), key=lambda x: -x[1])[:5]
        cluster_top[cid] = [name for name, _ in top5]

    print("\n[ 군집별 상위 5 활동 ]")
    for cid, tops in cluster_top.items():
        print(f"  C{cid+1}: {tops}")

    # --- 제외 키워드 (신체활동 포함 군집을 제외) ---
    EXCLUDE_KEYWORDS = [
        "신체활동", "물건 조종", "기계장치 제어", "차량, 기계, 장비 작동",
        "기계장비 유지 보수", "기계장치", "물건조종", "기계제어",
    ]

    def has_exclude(top_list):
        for item in top_list:
            for kw in EXCLUDE_KEYWORDS:
                if kw in item:
                    return True
        return False

    excluded = [cid for cid, tops in cluster_top.items() if has_exclude(tops)]
    included = [cid for cid in range(K) if cid not in excluded]

    print(f"\n제외 군집 (신체/기계): {['C'+str(c+1) for c in excluded]}")
    print(f"시뮬레이션 대상 군집: {['C'+str(c+1) for c in included]}")

    # 5개 이상の場合は上位5つに絞る (Silhouette 上位を維持)
    if len(included) > 5:
        # 각 군집의 내부 응집도(inertia 대리) 로 정렬해 상위 5 선정
        cohesion = {}
        for cid in included:
            mask   = labels == cid
            center = X[mask].mean(axis=0)
            dists  = np.linalg.norm(X[mask] - center, axis=1)
            cohesion[cid] = dists.mean()
        included = sorted(included, key=lambda c: cohesion[c])[:5]
        print(f"→ 상위 5 군집으로 축소: {['C'+str(c+1) for c in included]}")
    elif len(included) < 5:
        # 제외 군집에서 키워드 약한 것을 추가
        candidates = sorted(excluded, key=lambda c: sum(
            any(kw in item for kw in EXCLUDE_KEYWORDS[:3])
            for item in cluster_top[c]
        ))
        need = 5 - len(included)
        included += candidates[:need]
        print(f"→ {need}개 추가해 5개 확보: {['C'+str(c+1) for c in included]}")

    # 최종 5개 군집 ID (1-indexed 文字列 "C1"~"C8")
    sim_set = set(included)

    # --- results DataFrame 작성 ---
    result_rows = []
    for code, label, name in zip(df.index, labels, job_names):
        cid   = label   # 0-indexed
        label_str = f"C{cid + 1}"
        top_act   = "|".join(cluster_top[cid])
        is_target = 1 if cid in sim_set else 0
        result_rows.append({
            "job_code":            code,
            "job_name":            name,
            "cluster_id":          label_str,
            "top_activities":      top_act,
            "is_simulation_target": is_target,
        })

    df_out = pd.DataFrame(result_rows)
    out_path = os.path.join(PROC_DIR, "clusters.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n[02] clusters.csv 저장 → {out_path}")
    print(df_out[df_out["is_simulation_target"] == 1][["cluster_id", "job_name"]].head(20))

    # --- 군집 요약 출력 ---
    print("\n[ 시뮬레이션 대상 군집 요약 ]")
    for cid in sorted(sim_set):
        cnt  = (labels == cid).sum()
        tops = cluster_top[cid]
        print(f"  C{cid+1} ({cnt}개 직업) 상위활동: {tops[:3]}")


if __name__ == "__main__":
    main()
