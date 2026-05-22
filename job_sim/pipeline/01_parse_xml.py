"""
01_parse_xml.py
data/raw/ 以下のサブディレクトリから dtlGb_5.xml (능력), dtlGb_6.xml (성격),
dtlGb_7.xml (업무활동) をパースして職業×項目マトリクス3本を生成する。

産出物:
  data/processed/activities.csv    (41種)
  data/processed/abilities.csv     (44種)
  data/processed/personalities.csv (16種)
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# --- パス設定 ---
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# --- パース関数 ---

def parse_job_dirs(raw_dir):
    """raw_dir 内の全 K000xxxxxx ディレクトリを返す"""
    dirs = sorted([
        d for d in os.listdir(raw_dir)
        if d.startswith("K") and os.path.isdir(os.path.join(raw_dir, d))
    ])
    return dirs


def parse_activities(job_dir, job_code):
    """dtlGb_7.xml から업무활동(業務活動) をパース"""
    path = os.path.join(job_dir, "dtlGb_7.xml")
    if not os.path.exists(path):
        return None, None, None

    tree = ET.parse(path)
    root = tree.getroot()

    job_name = root.findtext("jobSmclNm") or root.findtext("jobMdclNm") or ""
    records = {}
    for item in root.iter("jobActvImprtncCmpr"):
        name_el = item.find("jobActvImprtncNmCmpr")
        val_el  = item.find("jobActvImprtncStatusCmpr")
        if name_el is not None and val_el is not None:
            try:
                records[name_el.text.strip()] = float(val_el.text)
            except (ValueError, TypeError):
                pass
    return job_code, job_name, records


def parse_abilities(job_dir, job_code):
    """dtlGb_5.xml から능력(能力) をパース"""
    path = os.path.join(job_dir, "dtlGb_5.xml")
    if not os.path.exists(path):
        return None, None, None

    tree = ET.parse(path)
    root = tree.getroot()

    job_name = root.findtext("jobSmclNm") or root.findtext("jobMdclNm") or ""
    records = {}
    for item in root.iter("jobAbilCmpr"):
        name_el = item.find("jobAblNmCmpr")
        val_el  = item.find("jobAblStatusCmpr")
        if name_el is not None and val_el is not None:
            try:
                records[name_el.text.strip()] = float(val_el.text)
            except (ValueError, TypeError):
                pass
    return job_code, job_name, records


def parse_personalities(job_dir, job_code):
    """dtlGb_6.xml から성격(性格) をパース"""
    path = os.path.join(job_dir, "dtlGb_6.xml")
    if not os.path.exists(path):
        return None, None, None

    tree = ET.parse(path)
    root = tree.getroot()

    job_name = root.findtext("jobSmclNm") or root.findtext("jobMdclNm") or ""
    records = {}
    for item in root.iter("jobChrCmpr"):
        name_el = item.find("jobChrNmCmpr")
        val_el  = item.find("jobChrStatusCmpr")
        if name_el is not None and val_el is not None:
            try:
                records[name_el.text.strip()] = float(val_el.text)
            except (ValueError, TypeError):
                pass
    return job_code, job_name, records


def build_matrix(parse_fn, job_dirs, raw_dir, label):
    """職業×項目マトリクスを構築し、欠損値を平均で補完して返す"""
    rows = {}
    names = {}
    all_cols = set()

    for d in job_dirs:
        job_dir  = os.path.join(raw_dir, d)
        code, name, records = parse_fn(job_dir, d)
        if code is None:
            continue
        rows[code]  = records
        names[code] = name
        all_cols.update(records.keys())

    cols = sorted(all_cols)
    df = pd.DataFrame(index=sorted(rows.keys()), columns=cols, dtype=float)
    df.index.name   = "job_code"
    df.columns.name = None

    for code, records in rows.items():
        for col in cols:
            df.at[code, col] = records.get(col, np.nan)

    # 欠損値を列平均で補完
    before = df.isna().sum().sum()
    df = df.fillna(df.mean())
    after  = df.isna().sum().sum()

    # job_name 列を先頭に追加
    name_ser = pd.Series(names, name="job_name")
    df.insert(0, "job_name", name_ser)

    print(f"[{label}] 職業数={len(df)}, 項目数={len(cols)}, "
          f"欠損補完={before - after} セル")
    return df


def main():
    job_dirs = parse_job_dirs(RAW_DIR)
    print(f"職業フォルダ数: {len(job_dirs)}")

    # --- 業務活動 (41종) ---
    df_act = build_matrix(parse_activities, job_dirs, RAW_DIR, "activities")
    out_act = os.path.join(PROC_DIR, "activities.csv")
    df_act.to_csv(out_act, encoding="utf-8-sig")
    print(f"  → {out_act}  shape={df_act.shape}")

    # --- 能力 (44종) ---
    df_abl = build_matrix(parse_abilities, job_dirs, RAW_DIR, "abilities")
    out_abl = os.path.join(PROC_DIR, "abilities.csv")
    df_abl.to_csv(out_abl, encoding="utf-8-sig")
    print(f"  → {out_abl}  shape={df_abl.shape}")

    # --- 性格 (16종) ---
    df_per = build_matrix(parse_personalities, job_dirs, RAW_DIR, "personalities")
    out_per = os.path.join(PROC_DIR, "personalities.csv")
    df_per.to_csv(out_per, encoding="utf-8-sig")
    print(f"  → {out_per}  shape={df_per.shape}")

    print("\n[01] 완료 — data/processed/ に3件のCSVを書き出しました。")


if __name__ == "__main__":
    main()
