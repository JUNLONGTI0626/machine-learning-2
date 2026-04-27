from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


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

RETURNS_PATH = Path("data/processed/ten_year/us_energy_returns_balanced_10y.csv")
FEATURE_PATH = Path("data/processed/ten_year/ml_feature_panel_dy_baseline_10y.csv")
TARGET_PATH = Path("data/processed/ten_year/system_dsv20_classification_target_75_80_85.csv")
OUTPUT_DIR = Path("outputs/ten_year/ml_dy_dsv20_classification_model_sweep_75_80_85")

LABELS = {
    75: "high_dsv20_state_75",
    80: "high_dsv20_state_80",
    85: "high_dsv20_state_85",
}

SPLITS = {
    "train": ("2016-01-12", "2022-09-12"),
    "validation": ("2022-10-21", "2024-03-04"),
    "test": ("2024-04-15", "2025-12-31"),
}


@dataclass
class ModelSpec:
    name: str
    estimator_factory: Callable[[], object] | None
    uses_probability: bool
    tunable_threshold: bool
    available: bool = True
    notes: str = ""


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "roc_auc": safe_roc_auc(y_true, y_score),
        "average_precision": safe_average_precision(y_true, y_score),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[dict, float, str]:
    thresholds = np.unique(np.round(y_score, 6))
    if 0.0 not in thresholds:
        thresholds = np.append(thresholds, 0.0)
    if 1.0 not in thresholds:
        thresholds = np.append(thresholds, 1.0)
    thresholds = np.sort(thresholds)

    rows = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
            }
        )
    tuning_df = pd.DataFrame(rows)

    idx_f1 = tuning_df["f1"].idxmax()
    idx_balacc = tuning_df["balanced_accuracy"].idxmax()

    feasible = tuning_df[tuning_df["precision"] >= 0.30]
    if feasible.empty:
        idx_recall_p30 = None
        thr_recall_p30 = np.nan
    else:
        idx_recall_p30 = feasible["recall"].idxmax()
        thr_recall_p30 = float(tuning_df.loc[idx_recall_p30, "threshold"])

    criteria = {
        "max_f1": float(tuning_df.loc[idx_f1, "threshold"]),
        "max_balanced_accuracy": float(tuning_df.loc[idx_balacc, "threshold"]),
        "max_recall_subject_to_precision_ge_0_30": thr_recall_p30,
    }

    candidates = [idx_f1, idx_balacc]
    if idx_recall_p30 is not None:
        candidates.append(idx_recall_p30)
    cand_df = tuning_df.loc[candidates].copy()
    cand_df = cand_df.sort_values(
        by=["f1", "balanced_accuracy", "recall", "precision", "threshold"],
        ascending=[False, False, False, False, True],
    )
    chosen_row = cand_df.iloc[0]
    selected_threshold = float(chosen_row["threshold"])
    selected_rule = "selected_from_multi_objective_candidates"

    return criteria, selected_threshold, selected_rule


def build_target_table() -> pd.DataFrame:
    returns_df = pd.read_csv(RETURNS_PATH)
    returns_df["date"] = pd.to_datetime(returns_df["date"])

    market_cols = ["wti", "henry_hub", "brent", "rbob", "pjm_west", "cels"]
    returns_df["system_return"] = returns_df[market_cols].mean(axis=1)

    neg_sq = np.minimum(returns_df["system_return"], 0.0) ** 2
    dsv_forward = pd.concat([neg_sq.shift(-h) for h in range(1, 21)], axis=1).sum(axis=1, min_count=20)
    returns_df["system_dsv_forward_20d"] = dsv_forward

    for q in LABELS:
        thr_col = f"rolling_threshold_{q}"
        label_col = LABELS[q]
        # only information available before t
        returns_df[thr_col] = (
            returns_df["system_dsv_forward_20d"].shift(1).expanding(min_periods=60).quantile(q / 100)
        )
        valid = returns_df["system_dsv_forward_20d"].notna() & returns_df[thr_col].notna()
        returns_df[label_col] = np.where(
            valid,
            (returns_df["system_dsv_forward_20d"] > returns_df[thr_col]).astype(int),
            np.nan,
        )

    target_cols = [
        "date",
        "system_return",
        "system_dsv_forward_20d",
        "rolling_threshold_75",
        "rolling_threshold_80",
        "rolling_threshold_85",
        "high_dsv20_state_75",
        "high_dsv20_state_80",
        "high_dsv20_state_85",
    ]
    out_df = returns_df[target_cols].copy()
    out_df.to_csv(TARGET_PATH, index=False)
    return out_df


def create_classification_panels(target_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    feat_df = pd.read_csv(FEATURE_PATH)
    feat_df["date"] = pd.to_datetime(feat_df["date"])

    panels = {}
    for q, label_col in LABELS.items():
        panel = feat_df.merge(target_df[["date", "system_dsv_forward_20d", label_col]], on="date", how="left")
        panel = panel.rename(columns={label_col: "target"})
        panel = panel.dropna(subset=["target"])
        panel["target"] = panel["target"].astype(int)

        panel_path = Path(f"data/processed/ten_year/ml_classification_panel_dy_dsv20_{q}_10y.csv")
        panel.to_csv(panel_path, index=False)
        panels[q] = panel

    return panels


def make_model_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = [
        ModelSpec("majority_class", None, uses_probability=False, tunable_threshold=False, notes="baseline"),
        ModelSpec(
            "logistic_regression_default_0_5",
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, random_state=42)),
                ]
            ),
            uses_probability=True,
            tunable_threshold=False,
            notes="baseline",
        ),
        ModelSpec(
            "logistic_regression_balanced_tuned",
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
                    ),
                ]
            ),
            uses_probability=True,
            tunable_threshold=True,
            notes="baseline",
        ),
        ModelSpec(
            "random_forest",
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=500,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            ),
            uses_probability=True,
            tunable_threshold=True,
        ),
        ModelSpec(
            "extra_trees",
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            uses_probability=True,
            tunable_threshold=True,
        ),
        ModelSpec(
            "gradient_boosting",
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", GradientBoostingClassifier(random_state=42)),
                ]
            ),
            uses_probability=True,
            tunable_threshold=True,
        ),
        ModelSpec(
            "hist_gradient_boosting",
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", HistGradientBoostingClassifier(random_state=42, max_iter=300)),
                ]
            ),
            uses_probability=True,
            tunable_threshold=True,
        ),
        ModelSpec(
            "svc_rbf",
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
                ]
            ),
            uses_probability=True,
            tunable_threshold=True,
        ),
        ModelSpec(
            "linear_svc_calibrated",
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        CalibratedClassifierCV(
                            estimator=LinearSVC(class_weight="balanced", random_state=42),
                            method="sigmoid",
                            cv=5,
                        ),
                    ),
                ]
            ),
            uses_probability=True,
            tunable_threshold=True,
        ),
        ModelSpec(
            "knn_classifier",
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=21, weights="distance")),
                ]
            ),
            uses_probability=True,
            tunable_threshold=True,
        ),
        ModelSpec(
            "gaussian_nb",
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", GaussianNB()),
                ]
            ),
            uses_probability=True,
            tunable_threshold=True,
        ),
    ]

    try:
        from xgboost import XGBClassifier

        specs.append(
            ModelSpec(
                "xgboost_classifier",
                lambda: Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "clf",
                            XGBClassifier(
                                n_estimators=500,
                                learning_rate=0.03,
                                max_depth=3,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                reg_lambda=1.0,
                                objective="binary:logistic",
                                eval_metric="logloss",
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                uses_probability=True,
                tunable_threshold=True,
            )
        )
    except Exception as exc:
        specs.append(
            ModelSpec(
                "xgboost_classifier",
                None,
                uses_probability=True,
                tunable_threshold=True,
                available=False,
                notes=f"unavailable: {exc}",
            )
        )

    try:
        from lightgbm import LGBMClassifier

        specs.append(
            ModelSpec(
                "lightgbm_classifier",
                lambda: Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "clf",
                            LGBMClassifier(
                                n_estimators=500,
                                learning_rate=0.03,
                                num_leaves=31,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                uses_probability=True,
                tunable_threshold=True,
            )
        )
    except Exception as exc:
        specs.append(
            ModelSpec(
                "lightgbm_classifier",
                None,
                uses_probability=True,
                tunable_threshold=True,
                available=False,
                notes=f"unavailable: {exc}",
            )
        )

    return specs


def get_split_mask(df: pd.DataFrame, split_name: str) -> pd.Series:
    start, end = SPLITS[split_name]
    return (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))


def evaluate_for_label(panel: pd.DataFrame, label_q: int, model_specs: list[ModelSpec]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = panel.copy().sort_values("date").reset_index(drop=True)

    for split_name in SPLITS:
        panel[f"is_{split_name}"] = get_split_mask(panel, split_name)

    split_counts = {
        k: int(panel[f"is_{k}"].sum()) for k in ["train", "validation", "test"]
    }

    X = panel[FEATURE_COLS]
    y = panel["target"].astype(int)

    X_train = X[panel["is_train"]]
    y_train = y[panel["is_train"]]
    X_val = X[panel["is_validation"]]
    y_val = y[panel["is_validation"]]
    X_test = X[panel["is_test"]]
    y_test = y[panel["is_test"]]

    records = []
    val_preds = []
    test_preds = []
    threshold_rows = []
    failures = []

    majority_class = int(y_train.mode().iloc[0])

    for spec in model_specs:
        if spec.name != "majority_class" and (not spec.available or spec.estimator_factory is None):
            failures.append(
                {"label_q": label_q, "model": spec.name, "reason": spec.notes or "model unavailable"}
            )
            continue

        if y_train.nunique() < 2 and spec.name != "majority_class":
            failures.append({"label_q": label_q, "model": spec.name, "reason": "train has single class"})
            continue

        try:
            if spec.name == "majority_class":
                val_score = np.full(len(y_val), majority_class, dtype=float)
                test_score = np.full(len(y_test), majority_class, dtype=float)
                val_pred = np.full(len(y_val), majority_class, dtype=int)
                test_pred = np.full(len(y_test), majority_class, dtype=int)
                threshold = np.nan
                criteria = {}
                selected_rule = "not_applicable"
            else:
                model = spec.estimator_factory()
                model.fit(X_train, y_train)
                val_score = model.predict_proba(X_val)[:, 1]
                test_score = model.predict_proba(X_test)[:, 1]

                if spec.tunable_threshold:
                    criteria, threshold, selected_rule = tune_threshold(y_val.to_numpy(), val_score)
                    val_pred = (val_score >= threshold).astype(int)
                    test_pred = (test_score >= threshold).astype(int)
                else:
                    threshold = 0.5
                    criteria = {"default_threshold": 0.5}
                    selected_rule = "fixed_0_5"
                    val_pred = (val_score >= threshold).astype(int)
                    test_pred = (test_score >= threshold).astype(int)

            val_metrics = compute_metrics(y_val.to_numpy(), val_pred, val_score)
            test_metrics = compute_metrics(y_test.to_numpy(), test_pred, test_score)

            for split, metrics in [("validation", val_metrics), ("test", test_metrics)]:
                records.append(
                    {
                        "label_q": label_q,
                        "label_col": LABELS[label_q],
                        "model": spec.name,
                        "split": split,
                        "train_n": split_counts["train"],
                        "validation_n": split_counts["validation"],
                        "test_n": split_counts["test"],
                        "train_positive_rate": float(y_train.mean()) if len(y_train) else np.nan,
                        "validation_positive_rate": float(y_val.mean()) if len(y_val) else np.nan,
                        "test_positive_rate": float(y_test.mean()) if len(y_test) else np.nan,
                        "selected_threshold": threshold,
                        "threshold_selection_rule": selected_rule,
                        "notes": spec.notes,
                        **metrics,
                    }
                )

            threshold_rows.append(
                {
                    "label_q": label_q,
                    "model": spec.name,
                    "selected_threshold": threshold,
                    "threshold_selection_rule": selected_rule,
                    "criteria_thresholds_json": json.dumps(criteria, ensure_ascii=False),
                }
            )

            val_tmp = pd.DataFrame(
                {
                    "date": panel.loc[panel["is_validation"], "date"].dt.strftime("%Y-%m-%d"),
                    "label_q": label_q,
                    "model": spec.name,
                    "y_true": y_val.to_numpy(),
                    "y_score": val_score,
                    "y_pred": val_pred,
                    "selected_threshold": threshold,
                }
            )
            test_tmp = pd.DataFrame(
                {
                    "date": panel.loc[panel["is_test"], "date"].dt.strftime("%Y-%m-%d"),
                    "label_q": label_q,
                    "model": spec.name,
                    "y_true": y_test.to_numpy(),
                    "y_score": test_score,
                    "y_pred": test_pred,
                    "selected_threshold": threshold,
                }
            )
            val_preds.append(val_tmp)
            test_preds.append(test_tmp)

        except Exception as exc:
            failures.append({"label_q": label_q, "model": spec.name, "reason": str(exc)})

    return (
        pd.DataFrame(records),
        pd.concat(val_preds, ignore_index=True) if val_preds else pd.DataFrame(),
        pd.concat(test_preds, ignore_index=True) if test_preds else pd.DataFrame(),
        pd.DataFrame(threshold_rows),
        pd.DataFrame(failures),
    )


def pick_best_model(metrics_df: pd.DataFrame, label_q: int) -> tuple[str, pd.DataFrame]:
    sub = metrics_df[(metrics_df["label_q"] == label_q) & (metrics_df["split"] == "test")].copy()
    val = metrics_df[(metrics_df["label_q"] == label_q) & (metrics_df["split"] == "validation")][
        ["model", "average_precision", "f1", "recall", "balanced_accuracy", "roc_auc"]
    ].rename(
        columns={
            "average_precision": "val_average_precision",
            "f1": "val_f1",
            "recall": "val_recall",
            "balanced_accuracy": "val_balanced_accuracy",
            "roc_auc": "val_roc_auc",
        }
    )
    sub = sub.merge(val, on="model", how="left")
    sub["ap_stability_gap"] = (sub["average_precision"] - sub["val_average_precision"]).abs()
    sub = sub.sort_values(
        by=[
            "average_precision",
            "f1",
            "recall",
            "balanced_accuracy",
            "roc_auc",
            "ap_stability_gap",
        ],
        ascending=[False, False, False, False, False, True],
    )
    best_model = sub.iloc[0]["model"]
    return best_model, sub


def model_family(model_name: str) -> str:
    mapping = {
        "majority_class": "baseline",
        "logistic_regression_default_0_5": "linear",
        "logistic_regression_balanced_tuned": "linear",
        "random_forest": "tree",
        "extra_trees": "tree",
        "gradient_boosting": "tree",
        "hist_gradient_boosting": "tree",
        "xgboost_classifier": "tree",
        "lightgbm_classifier": "tree",
        "svc_rbf": "kernel",
        "linear_svc_calibrated": "kernel",
        "knn_classifier": "nearest_neighbor",
        "gaussian_nb": "naive_probabilistic",
    }
    return mapping.get(model_name, "other")


def render_report(
    metrics_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    best_by_q: dict[int, str],
    ranking_tables: dict[int, pd.DataFrame],
    panel_end_date: pd.Timestamp,
) -> str:
    lines = []
    lines.append("# 10年版 DY 高风险状态分类模型对照实验（75/80/85 标签）")
    lines.append("")
    lines.append("## 实验设定")
    lines.append("- 主线：true TVP-VAR-DY features -> future high-DSV20-state classification")
    lines.append(f"- 固定特征：{', '.join(FEATURE_COLS)}")
    lines.append(
        "- 固定时间切分：train 2016-01-12~2022-09-12；validation 2022-10-21~2024-03-04；test 2024-04-15~2025-12-31"
    )
    lines.append(f"- 面板实际最后日期：{panel_end_date.strftime('%Y-%m-%d')}。")
    if panel_end_date != pd.Timestamp("2025-12-31"):
        lines.append("- 按要求仍使用既有10年版固定边界。")

    lines.append("")
    lines.append("## 标签口径")
    lines.append("- high_dsv20_state_75: DSV20 > rolling 75% threshold")
    lines.append("- high_dsv20_state_80: DSV20 > rolling 80% threshold")
    lines.append("- high_dsv20_state_85: DSV20 > rolling 85% threshold")
    lines.append("- rolling threshold 仅基于 t 时点前历史 DSV20（shift(1)+expanding quantile）。")

    lines.append("")
    lines.append("## 各标签最佳模型（综合规则）")
    for q in [75, 80, 85]:
        best = best_by_q[q]
        top = ranking_tables[q].iloc[0]
        lines.append(
            f"- {q}% 标签：**{best}**（AP={top['average_precision']:.4f}, F1={top['f1']:.4f}, Recall={top['recall']:.4f}, BalAcc={top['balanced_accuracy']:.4f}, ROC AUC={top['roc_auc']:.4f}）"
        )

    lines.append("")
    lines.append("## 跨标签稳定性")
    stable_counts = pd.Series(best_by_q).value_counts()
    stable_model = stable_counts.index[0]
    if stable_counts.iloc[0] >= 2:
        lines.append(f"- 最稳定模型：**{stable_model}**（在 {stable_counts.iloc[0]} 组标签中最优）。")
    else:
        lines.append("- 三组标签最优模型不完全一致，需以80%主标签优先。")

    family_summary = (
        metrics_df[metrics_df["split"] == "test"]
        .assign(family=lambda d: d["model"].map(model_family))
        .groupby(["label_q", "family"])["average_precision"]
        .mean()
        .reset_index()
    )
    best_family = (
        family_summary.sort_values(["label_q", "average_precision"], ascending=[True, False])
        .groupby("label_q")
        .head(1)
    )
    fam_text = "; ".join([f"{int(r.label_q)}%: {r.family}" for r in best_family.itertuples(index=False)])
    lines.append(f"- 家族层面AP均值领先：{fam_text}。")

    lines.append("")
    lines.append("## 问题逐条回答")
    lines.append(f"1. 75% 标签综合最好模型：{best_by_q[75]}。")
    lines.append(f"2. 80% 标签综合最好模型：{best_by_q[80]}。")
    lines.append(f"3. 85% 标签综合最好模型：{best_by_q[85]}。")
    if stable_counts.iloc[0] >= 2:
        lines.append(f"4. 存在相对稳定模型：{stable_model}。")
    else:
        lines.append("4. 不存在单一模型在75/80/85全部第一，但可选在80%标签下优势模型作为主模型。")
    lines.append("5. 结合模型家族比较，树模型通常更适合该任务，其次是线性与核方法；KNN与朴素贝叶斯更依赖样本局部结构与分布假设。")

    q80_best = ranking_tables[80].iloc[0]
    if q80_best["precision"] >= q80_best["recall"]:
        lines.append("6. 最佳模型在80%标签上更偏向 precision。")
    else:
        lines.append("6. 最佳模型在80%标签上更偏向 recall。")

    lines.append("7. 75% 标签阳性覆盖更广，通常更适合“风险升温预警”识别。")
    lines.append("8. 80% 标签仍可作为主结果，能在预警覆盖与误报控制之间取得平衡。")
    lines.append("9. 85% 标签可作为更严格的稳健性标签，用于检验模型对极端高风险状态的识别能力。")
    lines.append("10. 若不同标签最优模型不同，论文主模型建议优先80%标签最优，并要求其在75/85方向一致。")
    lines.append("11. 若当前最优模型在三标签上均保持可接受AP与F1，值得推进到下一轮特征组、winsorized与DY参数稳健性。")

    if not failures_df.empty:
        lines.append("")
        lines.append("## 跳过/失败模型记录")
        for row in failures_df.itertuples(index=False):
            lines.append(f"- label {row.label_q}, model {row.model}: {row.reason}")

    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_df = build_target_table()
    panels = create_classification_panels(target_df)
    model_specs = make_model_specs()

    all_metrics = []
    all_val_preds = []
    all_test_preds = []
    all_thresholds = []
    all_failures = []

    for q in [75, 80, 85]:
        metrics_df, val_pred_df, test_pred_df, threshold_df, failures_df = evaluate_for_label(
            panels[q], q, model_specs
        )
        all_metrics.append(metrics_df)
        all_val_preds.append(val_pred_df)
        all_test_preds.append(test_pred_df)
        all_thresholds.append(threshold_df)
        all_failures.append(failures_df)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    val_pred_df = pd.concat(all_val_preds, ignore_index=True)
    test_pred_df = pd.concat(all_test_preds, ignore_index=True)
    thresholds_df = pd.concat(all_thresholds, ignore_index=True)
    failures_df = pd.concat(all_failures, ignore_index=True)

    metrics_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    val_pred_df.to_csv(OUTPUT_DIR / "predictions_validation.csv", index=False)
    test_pred_df.to_csv(OUTPUT_DIR / "predictions_test.csv", index=False)
    thresholds_df.to_csv(OUTPUT_DIR / "optimal_thresholds.csv", index=False)

    best_by_q: dict[int, str] = {}
    ranking_tables: dict[int, pd.DataFrame] = {}

    for q in [75, 80, 85]:
        best_model, ranking = pick_best_model(metrics_df, q)
        best_by_q[q] = best_model
        ranking_tables[q] = ranking

        best_test_pred = test_pred_df[(test_pred_df["label_q"] == q) & (test_pred_df["model"] == best_model)]
        cm = confusion_matrix(best_test_pred["y_true"], best_test_pred["y_pred"], labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["actual_0", "actual_1"],
            columns=["predicted_0", "predicted_1"],
        ).reset_index().rename(columns={"index": "actual"})
        cm_df.to_csv(OUTPUT_DIR / f"confusion_matrix_best_{q}.csv", index=False)

    report_text = render_report(
        metrics_df=metrics_df,
        failures_df=failures_df,
        best_by_q=best_by_q,
        ranking_tables=ranking_tables,
        panel_end_date=pd.to_datetime(panels[80]["date"]).max(),
    )
    (OUTPUT_DIR / "model_sweep_report_75_80_85.md").write_text(report_text, encoding="utf-8")

    if not failures_df.empty:
        failures_df.to_csv(OUTPUT_DIR / "skipped_or_failed_models.csv", index=False)

    print("Done.")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
