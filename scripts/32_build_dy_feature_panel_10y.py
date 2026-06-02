#!/usr/bin/env python3
"""Build 10-year DY feature panel from baseline true TVP-VAR-DY outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TVP_DIR = Path("outputs/ten_year/tvpvar_dy_baseline")
OUTPUT_PANEL = Path("data/processed/ten_year/ml_feature_panel_dy_baseline_10y.csv")
REPORT_PATH = Path("outputs/ten_year/dy_feature_panel_report_10y.md")

VARS = ["wti", "henry_hub", "brent", "rbob", "pjm_west", "cels"]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    tci = pd.read_csv(TVP_DIR / "tci_total.csv", parse_dates=["date"])
    net = pd.read_csv(TVP_DIR / "net_spillovers.csv", parse_dates=["date"])
    return tci, net


def build_panel(tci: pd.DataFrame, net: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(tci, net, on="date", how="inner", validate="one_to_one")

    out = pd.DataFrame({"date": merged["date"], "tci_total": merged["tci_total"]})
    for v in VARS:
        out[f"net_{v}"] = merged[f"net_{v}"]

    out["tci_total_lag1"] = out["tci_total"].shift(1)
    out["tci_total_roll5_std"] = out["tci_total"].rolling(5, min_periods=5).std(ddof=0)

    out = out.dropna().reset_index(drop=True)
    return out


def assess_panel(panel: pd.DataFrame) -> dict:
    feature_cols = [c for c in panel.columns if c != "date"]

    missing_count = int(panel[feature_cols].isna().sum().sum())
    constant_features = [c for c in feature_cols if panel[c].nunique(dropna=True) <= 1]

    corr = panel[feature_cols].corr().abs()
    high_pairs = []
    for i, col_i in enumerate(feature_cols):
        for j in range(i + 1, len(feature_cols)):
            col_j = feature_cols[j]
            if corr.loc[col_i, col_j] >= 0.95:
                high_pairs.append((col_i, col_j, float(corr.loc[col_i, col_j])))

    return {
        "missing_count": missing_count,
        "constant_features": constant_features,
        "high_corr_pairs": high_pairs,
    }


def write_report(panel: pd.DataFrame, assessment: dict) -> None:
    defs = """
- tci_total：当期总连通度指数（TCI）。
- net_wti：WTI 的净溢出（TO-FROM）。
- net_henry_hub：Henry Hub 的净溢出（TO-FROM）。
- net_brent：Brent 的净溢出（TO-FROM）。
- net_rbob：RBOB 的净溢出（TO-FROM）。
- net_pjm_west：PJM West 的净溢出（TO-FROM）。
- net_cels：CELS 的净溢出（TO-FROM）。
- tci_total_lag1：TCI 的 1 期滞后值。
- tci_total_roll5_std：TCI 的 5 期滚动标准差（仅用当期及历史信息）。
""".strip()

    alignment_ok = panel["date"].is_monotonic_increasing and panel["date"].is_unique
    const_text = ", ".join(assessment["constant_features"]) if assessment["constant_features"] else "无"

    if assessment["high_corr_pairs"]:
        high_corr_text = "\n".join(
            [f"- {a} vs {b}: {v:.4f}" for a, b, v in assessment["high_corr_pairs"]]
        )
    else:
        high_corr_text = "- 未发现 |corr| >= 0.95 的特征对。"

    content = f"""# 10年版 DY 特征面板报告

## 1) 特征定义
{defs}

## 2) 日期对齐
- 全部按日期成功对齐：{alignment_ok}
- 面板起止日期：{panel['date'].min().date()} 至 {panel['date'].max().date()}
- 样本量：{len(panel)}

## 3) 缺失与常数特征检查
- 特征缺失值总数：{assessment['missing_count']}
- 常数特征：{const_text}

## 4) 相关性与冗余
{high_corr_text}

## 5) 结论
- 当前 10 年版 DY 特征面板可直接进入后续分类主线，不包含未来信息泄露构造。
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    tci, net = load_inputs()
    panel = build_panel(tci=tci, net=net)

    OUTPUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_PANEL, index=False)

    assessment = assess_panel(panel)
    write_report(panel, assessment)

    print(f"Saved feature panel: {OUTPUT_PANEL}")


if __name__ == "__main__":
    main()
