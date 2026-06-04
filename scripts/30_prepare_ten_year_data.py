#!/usr/bin/env python3
"""Prepare standalone ten_year data pipeline for US energy dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/ten_year")
INTERIM_OUT = Path("data/interim/ten_year/us_energy_levels_aligned_10y.csv")
RETURNS_OUT = Path("data/processed/ten_year/us_energy_returns_10y.csv")
BALANCED_OUT = Path("data/processed/ten_year/us_energy_returns_balanced_10y.csv")
REPORT_OUT = Path("outputs/ten_year/data_preparation_report_10y.md")

TARGET_VARS = ["wti", "henry_hub", "brent", "rbob", "pjm_west", "cels"]

FILE_NAME_MAP = {
    "wti": "wti",
    "henry": "henry_hub",
    "brent": "brent",
    "rbob": "rbob",
    "pjm": "pjm_west",
    "cels": "cels",
}

COLUMN_HINT_MAP = {
    "dcoilwtico": "wti",
    "dhhngsp": "henry_hub",
    "dcoilbrenteu": "brent",
    "drgasla": "rbob",
    "nasdaqcels": "cels",
    "pjm": "pjm_west",
}


@dataclass
class SeriesMeta:
    source_file: str
    variable: str
    rows_raw: int
    rows_after_clean: int
    duplicates_removed: int
    unparsed_values: int
    missing_values: int
    start_date: Optional[str]
    end_date: Optional[str]


def normalize_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_numeric(series: pd.Series) -> Tuple[pd.Series, int]:
    raw = series.astype(str)
    stripped = (
        raw.str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(r"[^0-9eE+\-.]", "", regex=True)
    )
    stripped = stripped.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
    parsed = pd.to_numeric(stripped, errors="coerce")
    unparsed = int((stripped.notna() & parsed.isna()).sum())
    return parsed, unparsed


def identify_variable(file_path: Path, columns: List[object]) -> str:
    lower_name = file_path.name.lower()
    for key, var in FILE_NAME_MAP.items():
        if key in lower_name:
            return var

    for col in columns:
        key = re.sub(r"[^a-z0-9]+", "", str(col).lower())
        for hint, var in COLUMN_HINT_MAP.items():
            if hint in key:
                return var

    raise ValueError(f"Cannot map file to target variable: {file_path}")


def parse_standard_file(file_path: Path, variable: str) -> Tuple[pd.DataFrame, int, int, int]:
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, dtype=str)
    else:
        df = pd.read_excel(file_path, dtype=str)

    rows_raw = len(df)
    cols = list(df.columns)
    date_col = None
    for col in cols:
        if "date" in str(col).lower() or "observation" in str(col).lower():
            date_col = col
            break
    if date_col is None:
        date_col = cols[0]

    candidate_cols = [c for c in cols if c != date_col]
    if not candidate_cols:
        raise ValueError(f"No value column found in {file_path}")

    value_col = candidate_cols[0]
    for c in candidate_cols:
        norm = re.sub(r"[^a-z0-9]+", "", str(c).lower())
        if variable == "wti" and "dcoilwtico" in norm:
            value_col = c
            break
        if variable == "henry_hub" and "dhhngsp" in norm:
            value_col = c
            break
        if variable == "brent" and "dcoilbrenteu" in norm:
            value_col = c
            break
        if variable == "rbob" and "drgasla" in norm:
            value_col = c
            break
        if variable == "cels" and "nasdaqcels" in norm:
            value_col = c
            break

    out = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"), variable: df[value_col]})
    out[variable], unparsed = clean_numeric(out[variable])
    out = out.dropna(subset=["date"])
    dup_count = int(out.duplicated(subset=["date"]).sum())
    out = out.sort_values("date").groupby("date", as_index=False).last()
    return out, rows_raw, dup_count, unparsed


def parse_pjm_file(file_path: Path) -> Tuple[pd.DataFrame, int, int, int]:
    df = pd.read_excel(file_path, header=None)
    rows_raw = len(df)

    # Source layout: col 3 = market date, col 6 = average peak price.
    date_series = pd.to_datetime(df.iloc[:, 3], errors="coerce")
    value_series = df.iloc[:, 6]

    out = pd.DataFrame({"date": date_series, "pjm_west": value_series})
    out["pjm_west"], unparsed = clean_numeric(out["pjm_west"])
    out = out.dropna(subset=["date"])
    dup_count = int(out.duplicated(subset=["date"]).sum())
    out = out.sort_values("date").groupby("date", as_index=False).last()
    return out, rows_raw, dup_count, unparsed


def raw_span(df: pd.DataFrame, var: str) -> Tuple[Optional[str], Optional[str], int]:
    valid = df.dropna(subset=[var])
    if valid.empty:
        return None, None, int(df[var].isna().sum())
    return (
        valid["date"].min().strftime("%Y-%m-%d"),
        valid["date"].max().strftime("%Y-%m-%d"),
        int(df[var].isna().sum()),
    )


def detect_anomalies(ret_df: pd.DataFrame) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for var in TARGET_VARS:
        s = ret_df[var]
        if s.notna().sum() < 30:
            out[var] = 0
            continue
        z = (s - s.mean()) / s.std(ddof=0)
        out[var] = int((z.abs() > 5).sum())
    return out


def format_date(date_val: pd.Timestamp) -> str:
    return date_val.strftime("%Y-%m-%d")


def safe_log_return(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.notna() & series.shift(1).notna() & (series > 0) & (series.shift(1) > 0)
    out.loc[valid] = np.log(series.loc[valid]) - np.log(series.shift(1).loc[valid])
    return out


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory not found: {RAW_DIR}")

    files = sorted([p for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}])
    if not files:
        raise RuntimeError(f"No supported raw files found in {RAW_DIR}")

    series_map: Dict[str, pd.DataFrame] = {}
    metas: List[SeriesMeta] = []
    scan_lines: List[str] = []

    for file_path in files:
        sample_cols: List[object]
        if file_path.suffix.lower() == ".csv":
            sample = pd.read_csv(file_path, nrows=1)
            sample_cols = list(sample.columns)
        else:
            sample = pd.read_excel(file_path, nrows=1)
            sample_cols = list(sample.columns)

        var = identify_variable(file_path, sample_cols)

        if var == "pjm_west":
            cleaned, rows_raw, dup_count, unparsed = parse_pjm_file(file_path)
        else:
            cleaned, rows_raw, dup_count, unparsed = parse_standard_file(file_path, var)

        start, end, missing = raw_span(cleaned, var)
        metas.append(
            SeriesMeta(
                source_file=str(file_path),
                variable=var,
                rows_raw=rows_raw,
                rows_after_clean=len(cleaned),
                duplicates_removed=dup_count,
                unparsed_values=unparsed,
                missing_values=missing,
                start_date=start,
                end_date=end,
            )
        )
        series_map[var] = cleaned[["date", var]]
        scan_lines.append(f"- `{file_path}` -> `{var}`")

    missing_vars = [v for v in TARGET_VARS if v not in series_map]
    if missing_vars:
        raise RuntimeError(f"Missing required variables from ten_year raw data: {missing_vars}")

    aligned = series_map[TARGET_VARS[0]].copy()
    for var in TARGET_VARS[1:]:
        aligned = aligned.merge(series_map[var], on="date", how="outer")
    aligned = aligned.sort_values("date").reset_index(drop=True)

    INTERIM_OUT.parent.mkdir(parents=True, exist_ok=True)
    aligned_out = aligned.copy()
    aligned_out["date"] = aligned_out["date"].dt.strftime("%Y-%m-%d")
    aligned_out.to_csv(INTERIM_OUT, index=False)

    returns = aligned.copy()
    for var in ["wti", "henry_hub", "brent", "rbob", "cels"]:
        returns[var] = safe_log_return(returns[var])

    pjm_non_missing = returns["pjm_west"].dropna()
    pjm_has_non_positive = bool((pjm_non_missing <= 0).any())
    if pjm_has_non_positive:
        returns["pjm_west"] = np.arcsinh(returns["pjm_west"]) - np.arcsinh(returns["pjm_west"].shift(1))
        pjm_method = "asinh difference"
    else:
        returns["pjm_west"] = safe_log_return(returns["pjm_west"])
        pjm_method = "log return"

    RETURNS_OUT.parent.mkdir(parents=True, exist_ok=True)
    returns_out = returns.copy()
    returns_out["date"] = returns_out["date"].dt.strftime("%Y-%m-%d")
    returns_out.to_csv(RETURNS_OUT, index=False)

    balanced = returns.dropna(subset=TARGET_VARS).copy()
    balanced_out = balanced.copy()
    balanced_out["date"] = balanced_out["date"].dt.strftime("%Y-%m-%d")
    BALANCED_OUT.parent.mkdir(parents=True, exist_ok=True)
    balanced_out.to_csv(BALANCED_OUT, index=False)

    ret_counts = returns[TARGET_VARS].notna().sum().sort_values()
    limiting_variable = ret_counts.index[0]

    anomalies = detect_anomalies(returns)
    abnormal_vars = [f"{k}({v})" for k, v in anomalies.items() if v > 0]

    pjm_note = "未发现旧样本对照文件。"
    old_pjm = Path("data/raw/pjm_west_daily.xlsx")
    if old_pjm.exists():
        old_df = pd.read_excel(old_pjm, header=None)
        same_shape = old_df.shape[1] == pd.read_excel(RAW_DIR / "pjm_west_daily.xlsx", header=None).shape[1]
        pjm_note = "与旧样本结构一致（列数一致，采用同一口径列）" if same_shape else "与旧样本结构可能存在差异（列数不一致）"

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUT.open("w", encoding="utf-8") as f:
        f.write("# Ten-year data preparation report\n\n")
        f.write("## 1) Raw file scan and variable mapping\n")
        f.write("\n".join(scan_lines) + "\n\n")

        f.write("## 2) Cleaning summary by variable\n")
        f.write("| variable | source_file | raw_rows | rows_after_clean | duplicates_removed | unparsed_values | missing_values | start_date | end_date |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---|---|\n")
        for m in sorted(metas, key=lambda x: x.variable):
            f.write(
                f"| {m.variable} | {m.source_file} | {m.rows_raw} | {m.rows_after_clean} | {m.duplicates_removed} | {m.unparsed_values} | {m.missing_values} | {m.start_date or 'NA'} | {m.end_date or 'NA'} |\n"
            )
        f.write("\n")

        f.write("## 3) Return construction\n")
        f.write("- wti, henry_hub, brent, rbob, cels: log return\n")
        f.write(f"- pjm_west: {pjm_method}（检测到非正值: {'是' if pjm_has_non_positive else '否'}）\n\n")

        f.write("## 4) Balanced sample\n")
        if balanced.empty:
            f.write("- 共同完整样本为空，请检查数据覆盖与缺失。\n")
        else:
            f.write(f"- 起始日期: {format_date(balanced['date'].min())}\n")
            f.write(f"- 结束日期: {format_date(balanced['date'].max())}\n")
            f.write(f"- 观测数: {len(balanced)}\n")
        f.write(f"- 最限制样本长度变量: {limiting_variable}（returns 非缺失数最少）\n\n")

        f.write("## 5) Data quality notes\n")
        if abnormal_vars:
            f.write(f"- 检测到潜在异常收益（|z|>5）变量: {', '.join(abnormal_vars)}。\n")
        else:
            f.write("- 未检测到明显异常收益（|z|>5）。\n")
        f.write(f"- pjm_west 与旧样本口径检查: {pjm_note}。\n")

        ten_year_start = aligned["date"].min()
        ten_year_end = aligned["date"].max()
        span_days = (ten_year_end - ten_year_start).days if pd.notna(ten_year_start) and pd.notna(ten_year_end) else 0
        if span_days < 3650:
            f.write("- 注意：当前数据日历跨度不足 10 年，请在建模前确认样本覆盖。\n")

    print(f"Wrote aligned levels: {INTERIM_OUT}")
    print(f"Wrote returns: {RETURNS_OUT}")
    print(f"Wrote balanced returns: {BALANCED_OUT}")
    print(f"Wrote report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
