#!/usr/bin/env python3
"""Ingest, clean, align, and report U.S. energy market raw data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
INTERIM_OUT = Path("data/interim/us_energy_levels_aligned.csv")
PROCESSED_OUT = Path("data/processed/us_energy_returns.csv")
REPORT_OUT = Path("outputs/reports/data_cleaning_merge_report.md")

EXPECTED_VARS = ["wti", "henry_hub", "brent", "rbob", "pjm_west", "ercot_north", "cels"]

KEYWORD_MAP = {
    "wti": ["wti", "dcoilwtico", "west texas"],
    "henry_hub": ["henry", "dhhngsp", "natural gas"],
    "brent": ["brent", "dcoilbrenteu"],
    "rbob": ["rbob", "drgasla", "gasoline"],
    "pjm_west": ["pjm", "wh", "wtd avg price", "$/mwh"],
    "ercot_north": ["ercot", "north"],
    "cels": ["cels", "close/last", "close"],
}

DATE_COL_HINTS = ["date", "observation_date", "trade date", "timestamp", "time"]


@dataclass
class FileProcessResult:
    file_name: str
    variable: Optional[str]
    used_date_col: Optional[str]
    used_value_col: Optional[str]
    candidate_date_cols: List[str] = field(default_factory=list)
    candidate_numeric_cols: List[str] = field(default_factory=list)
    row_count_raw: int = 0
    row_count_clean: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    unresolved_reason: Optional[str] = None


def to_snake(name: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return re.sub(r"_+", "_", text)


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {path}")


def detect_variable(file_name: str, columns: List[str]) -> Optional[str]:
    joined = f"{file_name} {' '.join(columns)}".lower()
    hits = []
    for var, keywords in KEYWORD_MAP.items():
        score = sum(1 for k in keywords if k in joined)
        if score > 0:
            hits.append((var, score))
    if not hits:
        return None
    hits.sort(key=lambda x: x[1], reverse=True)
    if len(hits) > 1 and hits[0][1] == hits[1][1]:
        # Conservative handling when two labels tie.
        return None
    return hits[0][0]


def detect_date_col(df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
    cols = list(df.columns)
    cands = []
    for c in cols:
        c_l = str(c).strip().lower()
        if any(h in c_l for h in DATE_COL_HINTS):
            cands.append(c)
    return (cands[0] if cands else None), cands


def parse_numeric_series(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    txt = series.astype(str)
    issues = {
        "comma_removed": int(txt.str.contains(",", regex=False, na=False).sum()),
        "percent_removed": int(txt.str.contains("%", regex=False, na=False).sum()),
        "space_trimmed": int((txt != txt.str.strip()).sum()),
        "symbol_removed": int(txt.str.contains(r"[^0-9eE+\-\.,%\s]", regex=True, na=False).sum()),
    }
    cleaned = txt.str.strip()
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.str.replace(r"[^0-9eE+\-\.]", "", regex=True)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    return numeric, issues


def candidate_numeric_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cands: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        numeric, _ = parse_numeric_series(df[c])
        ratio = numeric.notna().mean() if len(df) else 0
        if ratio >= 0.6:
            cands.append(c)
    return cands


def pick_value_col(df: pd.DataFrame, variable: Optional[str], numeric_cands: List[str]) -> Optional[str]:
    if not numeric_cands:
        return None
    if variable == "cels":
        for c in numeric_cands:
            if "close" in str(c).lower():
                return c
    if variable == "pjm_west":
        for c in numeric_cands:
            if "wtd avg" in str(c).lower() or "avg" in str(c).lower():
                return c
    # If exactly one candidate, conservative and safe.
    if len(numeric_cands) == 1:
        return numeric_cands[0]
    # Prefer columns that look like price/close/value/index.
    preferred = [c for c in numeric_cands if re.search(r"price|close|last|settle|index", str(c), flags=re.I)]
    if len(preferred) == 1:
        return preferred[0]
    return None


def process_file(path: Path) -> Tuple[Optional[pd.DataFrame], FileProcessResult]:
    df = load_table(path)
    result = FileProcessResult(file_name=path.name, variable=None, used_date_col=None, used_value_col=None)
    result.row_count_raw = len(df)

    original_columns = list(df.columns)
    var = detect_variable(path.name, [str(c) for c in original_columns])
    result.variable = var

    date_col, date_candidates = detect_date_col(df)
    result.candidate_date_cols = [str(c) for c in date_candidates]
    if date_col is None:
        result.unresolved_reason = "No date-like column detected"
        return None, result

    num_cands = candidate_numeric_cols(df, exclude=[date_col])
    result.candidate_numeric_cols = [str(c) for c in num_cands]
    value_col = pick_value_col(df, var, num_cands)

    if var is None:
        if len(num_cands) == 1:
            var = to_snake(Path(path).stem)
            result.variable = var
            result.issues.append("Variable name inferred from filename due to unknown mapping")
        else:
            result.unresolved_reason = "Variable mapping ambiguous"
            return None, result

    if value_col is None:
        result.unresolved_reason = "Value column ambiguous or unavailable"
        return None, result

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", var]

    parsed_date = pd.to_datetime(out["date"], errors="coerce")
    bad_dates = int(parsed_date.isna().sum())
    if bad_dates > 0:
        result.issues.append(f"Invalid date rows dropped: {bad_dates}")
    out["date"] = parsed_date
    out = out.dropna(subset=["date"])

    cleaned_values, num_issues = parse_numeric_series(out[var])
    out[var] = cleaned_values

    issue_messages = {
        "comma_removed": "Comma separators removed",
        "percent_removed": "Percent symbols removed",
        "space_trimmed": "Leading/trailing spaces trimmed",
        "symbol_removed": "Text/symbol noise removed",
    }
    for key, count in num_issues.items():
        if count > 0:
            result.issues.append(f"{issue_messages[key]}: {count} rows")

    dup_count = int(out.duplicated(subset=["date"]).sum())
    if dup_count > 0:
        result.issues.append(f"Duplicate dates dropped (keep last): {dup_count}")
    out = out.drop_duplicates(subset=["date"], keep="last")

    # Ensure ascending date order.
    if not out["date"].is_monotonic_increasing:
        result.issues.append("Date order normalized to ascending")
    out = out.sort_values("date").reset_index(drop=True)

    result.used_date_col = str(date_col)
    result.used_value_col = str(value_col)
    result.row_count_clean = len(out)
    if len(out):
        result.start_date = out["date"].min().date().isoformat()
        result.end_date = out["date"].max().date().isoformat()

    return out, result


def compute_log_returns(level_df: pd.DataFrame) -> pd.DataFrame:
    ret = level_df.copy()
    numeric_cols = [c for c in ret.columns if c != "date"]
    for c in numeric_cols:
        s = pd.to_numeric(ret[c], errors="coerce")
        s = s.where(s > 0)  # log only for positive levels
        ret[c] = np.log(s).diff()
    return ret


def ensure_parents(*paths: Path) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


def write_report(results: List[FileProcessResult], merged: pd.DataFrame) -> None:
    identified = [r for r in results if r.unresolved_reason is None]
    unresolved = [r for r in results if r.unresolved_reason is not None]
    missing = merged.drop(columns=["date"]).isna().sum().sort_values(ascending=False)

    lines: List[str] = []
    lines.append("# Data Cleaning and Merge Report\n")
    lines.append("## 1) Files scanned and variable identification\n")
    for r in results:
        lines.append(f"- **{r.file_name}**")
        lines.append(f"  - mapped variable: `{r.variable}`")
        lines.append(f"  - detected date column: `{r.used_date_col}`")
        lines.append(f"  - selected value column: `{r.used_value_col}`")
        if r.candidate_numeric_cols:
            lines.append(f"  - numeric candidates: {', '.join([f'`{c}`' for c in r.candidate_numeric_cols])}")
        if r.unresolved_reason:
            lines.append(f"  - status: unresolved ({r.unresolved_reason})")
        else:
            lines.append("  - status: included")

    lines.append("\n## 2) Original sample ranges by variable\n")
    if identified:
        lines.append("| variable | file | start_date | end_date | raw_rows | clean_rows |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for r in identified:
            lines.append(
                f"| {r.variable} | {r.file_name} | {r.start_date} | {r.end_date} | {r.row_count_raw} | {r.row_count_clean} |"
            )
    else:
        lines.append("No variables identified.")

    lines.append("\n## 3) Cleaning actions taken\n")
    for r in identified:
        if r.issues:
            lines.append(f"- **{r.variable}** ({r.file_name}): " + "; ".join(r.issues))
        else:
            lines.append(f"- **{r.variable}** ({r.file_name}): no special issues detected")

    if unresolved:
        lines.append("\n## 4) Conservative handling of unresolved files\n")
        for r in unresolved:
            lines.append(f"- **{r.file_name}**: {r.unresolved_reason}")
            if r.candidate_date_cols:
                lines.append(f"  - date candidates: {', '.join([f'`{c}`' for c in r.candidate_date_cols])}")
            if r.candidate_numeric_cols:
                lines.append(f"  - numeric candidates: {', '.join([f'`{c}`' for c in r.candidate_numeric_cols])}")

    lines.append("\n## 5) Missingness after outer merge\n")
    lines.append("| variable | missing_count | missing_ratio |")
    lines.append("|---|---:|---:|")
    total_n = max(len(merged), 1)
    for col, miss in missing.items():
        lines.append(f"| {col} | {int(miss)} | {miss/total_n:.2%} |")

    start = merged["date"].min().date().isoformat() if len(merged) else "NA"
    end = merged["date"].max().date().isoformat() if len(merged) else "NA"
    lines.append("\n## 6) Final aligned sample\n")
    lines.append(f"- sample start date: **{start}**")
    lines.append(f"- sample end date: **{end}**")
    lines.append(f"- total observations (daily index rows): **{len(merged)}**")

    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    raw_files = sorted(
        [p for p in RAW_DIR.glob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"}],
        key=lambda p: p.name.lower(),
    )
    if not raw_files:
        raise FileNotFoundError(f"No raw files found in {RAW_DIR}")

    cleaned_frames: List[pd.DataFrame] = []
    results: List[FileProcessResult] = []

    for path in raw_files:
        cleaned_df, result = process_file(path)
        results.append(result)
        if cleaned_df is not None:
            cleaned_frames.append(cleaned_df)

    if not cleaned_frames:
        raise RuntimeError("No usable files after processing")

    merged = cleaned_frames[0]
    for df in cleaned_frames[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    returns = compute_log_returns(merged)

    ensure_parents(INTERIM_OUT, PROCESSED_OUT, REPORT_OUT)
    merged.to_csv(INTERIM_OUT, index=False)
    returns.to_csv(PROCESSED_OUT, index=False)
    write_report(results, merged)

    print(f"Saved level data: {INTERIM_OUT}")
    print(f"Saved return data: {PROCESSED_OUT}")
    print(f"Saved report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
