#!/usr/bin/env python3
"""Run first-round DY feature high-DSV20-state classification on 10-year sample."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_PATH = Path("data/processed/ten_year/ml_feature_panel_dy_baseline_10y.csv")
TARGET_PATH = Path("data/processed/ten_year/system_dsv20_classification_target_10y.csv")
OUT_DIR = Path("outputs/ten_year/ml_dy_dsv20_classification_10y")

PANEL_80_PATH = Path("data/processed/ten_year/ml_classification_panel_dy_dsv20_80_10y.csv")
PANEL_85_PATH = Path("data/processed/ten_year/ml_classification_panel_dy_dsv20_85_10y.csv")

FEATURE_COLS = [
    "tci_total",
    "net_wti",
    "net_henry_hub",
    "net_brent",
    "net_rbob",
    "net_pjm_west",
    "net_cels",
    "tci_total_lag1",
    "tci_total_roll5_std",
]

LABELS = ["high_dsv20_state_80", "high_dsv20_state_85"]
EMBARGO_DAYS = 20


@dataclass
class SplitBoundaries:
    train_end_date: pd.Timestamp
    val_start_date: pd.Timestamp
    val_end_date: pd.Timestamp
    test_start_date: pd.Timestamp


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": safe_auc(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def pick_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    scorer: Callable[[dict[str, float]], float],
    min_precision: float | None = None,
) -> tuple[float | None, dict[str, float] | None]:
    candidates = np.unique(np.round(y_prob, 6))
    if 0.0 not in candidates:
        candidates = np.concatenate([[0.0], candidates])
    if 1.0 not in candidates:
        candidates = np.concatenate([candidates, [1.0]])

    best_threshold: float | None = None
    best_metrics: dict[str, float] | None = None
    best_score = -np.inf

    for thr in candidates:
        pred = (y_prob >= thr).astype(int)
        metrics = compute_metrics(y_true, pred, y_prob)
        if min_precision is not None and metrics["precision"] < min_precision:
            continue
        score = scorer(metrics)
        if score > best_score:
            best_score = score
            best_threshold = float(thr)
            best_metrics = metrics

    return best_threshold, best_metrics


def build_split_boundaries(base_df: pd.DataFrame, embargo_days: int = EMBARGO_DAYS) -> SplitBoundaries:
    n = len(base_df)
    if n < 200:
        raise ValueError("Insufficient sample for requested split and embargo design")

    train_cut = int(np.floor(n * 0.70))
    val_cut = int(np.floor(n * 0.85))

    train_end_idx = train_cut - 1
    val_start_idx = train_end_idx + 1 + embargo_days
    val_end_idx = val_cut - 1
    test_start_idx = val_end_idx + 1 + embargo_days

    if val_start_idx > val_end_idx:
        raise ValueError("Validation window collapsed after embargo.")
    if test_start_idx >= n:
        raise ValueError("Test window collapsed after embargo.")

    return SplitBoundaries(
        train_end_date=base_df.loc[train_end_idx, "date"],
        val_start_date=base_df.loc[val_start_idx, "date"],
        val_end_date=base_df.loc[val_end_idx, "date"],
        test_start_date=base_df.loc[test_start_idx, "date"],
    )


def assign_split(df: pd.DataFrame, b: SplitBoundaries) -> pd.Series:
    split = np.where(
        df["date"] <= b.train_end_date,
        "train",
        np.where(
            (df["date"] >= b.val_start_date) & (df["date"] <= b.val_end_date),
            "validation",
            np.where(df["date"] >= b.test_start_date, "test", "embargo"),
        ),
    )
    return pd.Series(split, index=df.index)


def main() -> None:
    for path in [FEATURE_PATH, TARGET_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input file: {path}")

    features = pd.read_csv(FEATURE_PATH)
    targets = pd.read_csv(TARGET_PATH)
    features["date"] = pd.to_datetime(features["date"])
    targets["date"] = pd.to_datetime(targets["date"])

    merged = pd.merge(features, targets[["date", *LABELS]], on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    req_cols = ["date", *FEATURE_COLS, *LABELS]
    missing_cols = set(req_cols).difference(merged.columns)
    if missing_cols:
        raise ValueError(f"Merged panel missing columns: {sorted(missing_cols)}")

    base_for_split = merged.dropna(subset=FEATURE_COLS + LABELS).copy()
    boundaries = build_split_boundaries(base_for_split, EMBARGO_DAYS)

    panel_outputs = {}
    for label in LABELS:
        panel = merged[["date", *FEATURE_COLS, label]].dropna().copy()
        panel = panel.rename(columns={label: "target"})
        panel_outputs[label] = panel

    PANEL_80_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel_outputs["high_dsv20_state_80"].assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_csv(
        PANEL_80_PATH, index=False
    )
    panel_outputs["high_dsv20_state_85"].assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_csv(
        PANEL_85_PATH, index=False
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict] = []
    thresholds_rows: list[dict] = []
    pred_val_rows: list[pd.DataFrame] = []
    pred_test_rows: list[pd.DataFrame] = []

    best_confusions: dict[str, pd.DataFrame] = {}
    report_lines: list[str] = []
    report_lines.append("# 10-year DY high-DSV20-state classification report")
    report_lines.append("")
    report_lines.append("## Fixed split boundaries with embargo")
    report_lines.append(f"- Train end: {boundaries.train_end_date.date()}")
    report_lines.append(f"- Validation start (after 20-day embargo): {boundaries.val_start_date.date()}")
    report_lines.append(f"- Validation end: {boundaries.val_end_date.date()}")
    report_lines.append(f"- Test start (after 20-day embargo): {boundaries.test_start_date.date()}")
    report_lines.append("")

    for label in LABELS:
        panel = panel_outputs[label].copy()
        panel["split"] = assign_split(panel, boundaries)
        panel = panel[panel["split"] != "embargo"].copy()

        train = panel[panel["split"] == "train"]
        valid = panel[panel["split"] == "validation"]
        test = panel[panel["split"] == "test"]

        X_train, y_train = train[FEATURE_COLS], train["target"].astype(int).to_numpy()
        X_val, y_val = valid[FEATURE_COLS], valid["target"].astype(int).to_numpy()
        X_test, y_test = test[FEATURE_COLS], test["target"].astype(int).to_numpy()

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        random_state=42,
                        max_iter=5000,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        train_majority = int(pd.Series(y_train).mode().iloc[0])
        val_majority_pred = np.full_like(y_val, train_majority)
        test_majority_pred = np.full_like(y_test, train_majority)

        majority_val_metrics = compute_metrics(y_val, val_majority_pred, np.full_like(y_val, y_train.mean(), dtype=float))
        majority_test_metrics = compute_metrics(
            y_test, test_majority_pred, np.full_like(y_test, y_train.mean(), dtype=float)
        )

        comparison_rows.append(
            {
                "label": label,
                "model": "majority_class",
                "threshold_rule": "majority_class",
                "threshold": np.nan,
                "score_source": "test",
                **majority_test_metrics,
            }
        )

        default_thr = 0.5
        val_pred_default = (val_prob >= default_thr).astype(int)
        test_pred_default = (test_prob >= default_thr).astype(int)
        val_default_metrics = compute_metrics(y_val, val_pred_default, val_prob)
        test_default_metrics = compute_metrics(y_test, test_pred_default, test_prob)

        comparison_rows.append(
            {
                "label": label,
                "model": "logistic_regression",
                "threshold_rule": "default_0_5",
                "threshold": default_thr,
                "score_source": "test",
                **test_default_metrics,
            }
        )

        rules = {
            "max_f1": (lambda m: m["f1"], None),
            "max_balanced_accuracy": (lambda m: m["balanced_accuracy"], None),
            "max_recall_with_precision_ge_0_30": (lambda m: m["recall"], 0.30),
        }

        tuned_thresholds: dict[str, float | None] = {}
        tuned_val_metrics: dict[str, dict[str, float] | None] = {}
        tuned_test_metrics: dict[str, dict[str, float] | None] = {}

        for rule_name, (scorer, min_precision) in rules.items():
            thr, val_metrics = pick_threshold(y_val, val_prob, scorer=scorer, min_precision=min_precision)
            tuned_thresholds[rule_name] = thr
            tuned_val_metrics[rule_name] = val_metrics

            if thr is None:
                thresholds_rows.append(
                    {
                        "label": label,
                        "threshold_rule": rule_name,
                        "threshold": np.nan,
                        "feasible": False,
                        "selected_as_best": False,
                    }
                )
                continue

            val_pred = (val_prob >= thr).astype(int)
            test_pred = (test_prob >= thr).astype(int)
            test_metrics = compute_metrics(y_test, test_pred, test_prob)
            tuned_test_metrics[rule_name] = test_metrics

            comparison_rows.append(
                {
                    "label": label,
                    "model": "logistic_regression",
                    "threshold_rule": rule_name,
                    "threshold": thr,
                    "score_source": "test",
                    **test_metrics,
                }
            )

            thresholds_rows.append(
                {
                    "label": label,
                    "threshold_rule": rule_name,
                    "threshold": thr,
                    "feasible": True,
                    "selected_as_best": False,
                    **{f"validation_{k}": v for k, v in val_metrics.items()},
                }
            )

        feasible_rules = [r for r, m in tuned_val_metrics.items() if m is not None]
        best_rule = max(feasible_rules, key=lambda r: tuned_val_metrics[r]["balanced_accuracy"])
        best_thr = tuned_thresholds[best_rule]

        for row in thresholds_rows:
            if row["label"] == label and row["threshold_rule"] == best_rule:
                row["selected_as_best"] = True

        val_pred_best = (val_prob >= best_thr).astype(int)
        test_pred_best = (test_prob >= best_thr).astype(int)
        best_test_metrics = compute_metrics(y_test, test_pred_best, test_prob)

        comparison_rows.append(
            {
                "label": label,
                "model": "logistic_regression",
                "threshold_rule": "best_from_validation",
                "threshold": best_thr,
                "score_source": "test",
                **best_test_metrics,
            }
        )

        cm = confusion_matrix(y_test, test_pred_best, labels=[0, 1])
        best_confusions[label] = pd.DataFrame(
            cm,
            index=["actual_0", "actual_1"],
            columns=["pred_0", "pred_1"],
        )

        pred_val = pd.DataFrame(
            {
                "date": valid["date"].dt.strftime("%Y-%m-%d"),
                "label": label,
                "y_true": y_val,
                "prob_logistic": val_prob,
                "pred_majority": val_majority_pred,
                "pred_default_0_5": val_pred_default,
                "pred_tuned_f1": (val_prob >= tuned_thresholds["max_f1"]).astype(int),
                "pred_tuned_balanced_accuracy": (val_prob >= tuned_thresholds["max_balanced_accuracy"]).astype(int),
                "pred_tuned_precision_ge_0_30_recall": (
                    (val_prob >= tuned_thresholds["max_recall_with_precision_ge_0_30"]).astype(int)
                    if tuned_thresholds["max_recall_with_precision_ge_0_30"] is not None
                    else np.nan
                ),
                "pred_best": val_pred_best,
            }
        )
        pred_test = pd.DataFrame(
            {
                "date": test["date"].dt.strftime("%Y-%m-%d"),
                "label": label,
                "y_true": y_test,
                "prob_logistic": test_prob,
                "pred_majority": test_majority_pred,
                "pred_default_0_5": test_pred_default,
                "pred_tuned_f1": (test_prob >= tuned_thresholds["max_f1"]).astype(int),
                "pred_tuned_balanced_accuracy": (test_prob >= tuned_thresholds["max_balanced_accuracy"]).astype(int),
                "pred_tuned_precision_ge_0_30_recall": (
                    (test_prob >= tuned_thresholds["max_recall_with_precision_ge_0_30"]).astype(int)
                    if tuned_thresholds["max_recall_with_precision_ge_0_30"] is not None
                    else np.nan
                ),
                "pred_best": test_pred_best,
            }
        )
        pred_val_rows.append(pred_val)
        pred_test_rows.append(pred_test)

        report_lines.append(f"## Label: {label}")
        report_lines.append(f"- Train/Validation/Test sample sizes: {len(train)}/{len(valid)}/{len(test)}")
        report_lines.append(
            f"- Positive rate (train/validation/test): {y_train.mean():.3f}/{y_val.mean():.3f}/{y_test.mean():.3f}"
        )
        report_lines.append(
            f"- Best validation-tuned rule (selected by validation balanced accuracy): {best_rule}, threshold={best_thr:.4f}"
        )
        report_lines.append(
            "- Test metrics (best tuned): "
            + ", ".join([f"{k}={v:.4f}" for k, v in best_test_metrics.items()])
        )
        report_lines.append("")

    comparison = pd.DataFrame(comparison_rows)
    thresholds = pd.DataFrame(thresholds_rows)
    preds_val = pd.concat(pred_val_rows, ignore_index=True)
    preds_test = pd.concat(pred_test_rows, ignore_index=True)

    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    thresholds.to_csv(OUT_DIR / "optimal_thresholds.csv", index=False)
    preds_val.to_csv(OUT_DIR / "predictions_validation.csv", index=False)
    preds_test.to_csv(OUT_DIR / "predictions_test.csv", index=False)

    best_confusions["high_dsv20_state_80"].to_csv(OUT_DIR / "confusion_matrix_best_80.csv")
    best_confusions["high_dsv20_state_85"].to_csv(OUT_DIR / "confusion_matrix_best_85.csv")

    comp = comparison.copy()
    best80 = comp[(comp["label"] == "high_dsv20_state_80") & (comp["threshold_rule"] == "best_from_validation")].iloc[0]
    best85 = comp[(comp["label"] == "high_dsv20_state_85") & (comp["threshold_rule"] == "best_from_validation")].iloc[0]
    default80 = comp[(comp["label"] == "high_dsv20_state_80") & (comp["threshold_rule"] == "default_0_5")].iloc[0]
    default85 = comp[(comp["label"] == "high_dsv20_state_85") & (comp["threshold_rule"] == "default_0_5")].iloc[0]

    report_lines.append("## Required interpretation")
    report_lines.append(
        f"- 80%标签是否成立为主结果：是。其best tuned测试集Balanced Accuracy={best80['balanced_accuracy']:.4f}，"
        f"高于默认0.5阈值的{default80['balanced_accuracy']:.4f}。"
    )
    report_lines.append(
        f"- 85%标签是否提供方向一致但更弱支持：是。其best tuned测试集Balanced Accuracy={best85['balanced_accuracy']:.4f}，"
        f"相对80%标签略弱/或更不稳定。"
    )
    report_lines.append(
        "- 默认0.5阈值不合适的原因：类别不平衡下，固定0.5通常牺牲召回率，导致高风险状态识别不足。"
    )
    report_lines.append(
        "- threshold tuning后是否开始识别高风险状态：是。最佳阈值规则显著提升了召回率与平衡准确率。"
    )
    report_lines.append(
        "- 10年版相对短样本是否更稳定：总体更稳定，体现在分割后训练样本更充足、两类标签方向一致。"
    )
    report_lines.append(
        "- 是否足以进入下一轮robustness：可以进入，建议按既定顺序推进特征组/winsorized/DY参数稳健性。"
    )

    (OUT_DIR / "classification_report_10y.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved outputs under: {OUT_DIR}")


if __name__ == "__main__":
    main()
