#!/usr/bin/env python3
"""Run 10-year DY parameter robustness checks for fixed RF classification specs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


VARIABLES = ["wti", "henry_hub", "brent", "rbob", "pjm_west", "cels"]
BASELINE_PARAMS = {
    "p": 5,
    "H": 20,
    "discount_factor": 0.99,
    "innovation_covariance_smoothing": 0.96,
    "initialization_window": 250,
}

PARAMETER_VERSIONS = [
    {
        "version_id": "baseline",
        "label": "Baseline (p=5, H=20, discount=0.99)",
        **BASELINE_PARAMS,
    },
    {
        "version_id": "lag_p2",
        "label": "Lag robustness (p=2)",
        **{**BASELINE_PARAMS, "p": 2},
    },
    {
        "version_id": "horizon_h10",
        "label": "FEVD horizon robustness (H=10)",
        **{**BASELINE_PARAMS, "H": 10},
    },
    {
        "version_id": "discount_098",
        "label": "Discount robustness (discount=0.98)",
        **{**BASELINE_PARAMS, "discount_factor": 0.98},
    },
]

SPLITS = {
    "train": (pd.Timestamp("2016-01-12"), pd.Timestamp("2022-09-12")),
    "validation": (pd.Timestamp("2022-10-21"), pd.Timestamp("2024-03-04")),
    "test": (pd.Timestamp("2024-04-15"), pd.Timestamp("2025-11-21")),
}

SPECS = [
    {
        "spec_id": "spec1_main",
        "spec_name": "Spec 1 (80% + net-only)",
        "label_col": "high_dsv20_state_80",
        "features": [
            "net_wti",
            "net_henry_hub",
            "net_brent",
            "net_rbob",
            "net_pjm_west",
            "net_cels",
        ],
    },
    {
        "spec_id": "spec2_strict",
        "spec_name": "Spec 2 (85% + total-only)",
        "label_col": "high_dsv20_state_85",
        "features": ["tci_total", "tci_total_lag1", "tci_total_roll5_std"],
    },
]

THRESHOLD_RULES = [
    "default_0.5",
    "max_f1",
    "max_balanced_accuracy",
    "max_recall_precision_ge_0.30",
]


@dataclass
class TVPResult:
    dates: pd.Series
    tci_total: np.ndarray
    net_spillovers: np.ndarray


def _ols_init(y: np.ndarray, X: np.ndarray, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    xtx = X.T @ X + ridge * np.eye(X.shape[1])
    xty = X.T @ y
    beta = np.linalg.solve(xtx, xty)
    residual = y - X @ beta
    sigma2 = float(np.var(residual)) if residual.size > 1 else 1.0
    cov = np.linalg.inv(xtx) * max(sigma2, 1e-6)
    return beta, cov


def _companion_from_lags(a_mats: np.ndarray) -> np.ndarray:
    p, n, _ = a_mats.shape
    top = np.hstack([a_mats[lag] for lag in range(p)])
    if p == 1:
        return top
    bottom_left = np.eye(n * (p - 1))
    bottom_right = np.zeros((n * (p - 1), n))
    bottom = np.hstack([bottom_left, bottom_right])
    return np.vstack([top, bottom])


def _gfevd(a_mats: np.ndarray, sigma: np.ndarray, H: int) -> np.ndarray:
    p, n, _ = a_mats.shape
    if np.any(~np.isfinite(sigma)):
        sigma = np.nan_to_num(sigma, nan=0.0)
    sigma = (sigma + sigma.T) / 2.0
    sigma += 1e-8 * np.eye(n)

    F = _companion_from_lags(a_mats)
    J = np.hstack([np.eye(n), np.zeros((n, n * (p - 1)))]) if p > 1 else np.eye(n)

    theta_num = np.zeros((n, n))
    theta_den = np.zeros(n)
    Fh = np.eye(F.shape[0])

    for _ in range(H):
        phi_h = J @ Fh @ J.T
        phi_sigma = phi_h @ sigma
        for i in range(n):
            row_vec = phi_h[i, :]
            theta_den[i] += float(row_vec @ sigma @ row_vec.T)
            for j in range(n):
                num_ij = float(phi_sigma[i, j])
                theta_num[i, j] += (num_ij**2) / max(sigma[j, j], 1e-8)
        Fh = Fh @ F

    theta = np.divide(theta_num, theta_den[:, None] + 1e-12)
    row_sums = theta.sum(axis=1, keepdims=True)
    theta = np.divide(theta, row_sums + 1e-12)
    return theta


def run_tvp_var_dy(returns: pd.DataFrame, p: int, H: int, discount: float, cov_smoothing: float, init_window: int) -> TVPResult:
    y = returns[VARIABLES].to_numpy(dtype=float)
    dates = returns["date"].reset_index(drop=True)
    T, n = y.shape
    k = n * p

    if T <= init_window + p:
        raise ValueError("Not enough observations for initialization window and lag order.")

    X_all = []
    Y_all = []
    for t in range(p, T):
        x_t = np.concatenate([y[t - lag, :] for lag in range(1, p + 1)])
        X_all.append(x_t)
        Y_all.append(y[t, :])
    X_all = np.asarray(X_all)
    Y_all = np.asarray(Y_all)

    init_n = min(init_window, X_all.shape[0] - 1)
    betas = np.zeros((n, k))
    covs = np.zeros((n, k, k))
    for i in range(n):
        beta_i, cov_i = _ols_init(Y_all[:init_n, i], X_all[:init_n, :])
        betas[i, :] = beta_i
        covs[i, :, :] = cov_i

    sigma = np.cov((Y_all[:init_n, :] - X_all[:init_n, :] @ betas.T).T)
    sigma = np.nan_to_num(sigma, nan=0.0)
    sigma += 1e-6 * np.eye(n)

    tci_vals = []
    net_vals = []
    out_dates = []

    for idx in range(init_n, X_all.shape[0]):
        x_t = X_all[idx, :]
        y_t = Y_all[idx, :]
        pred = betas @ x_t
        resid = y_t - pred

        for i in range(n):
            P = covs[i, :, :]
            xcol = x_t[:, None]
            denom = discount + float(x_t @ P @ x_t)
            K = (P @ xcol) / max(denom, 1e-12)
            err = resid[i]
            betas[i, :] = betas[i, :] + (K[:, 0] * err)
            covs[i, :, :] = (P - K @ xcol.T @ P) / discount
            covs[i, :, :] = (covs[i, :, :] + covs[i, :, :].T) / 2.0

        sigma = cov_smoothing * sigma + (1.0 - cov_smoothing) * np.outer(resid, resid)
        sigma = (sigma + sigma.T) / 2.0

        a_mats = np.zeros((p, n, n))
        for lag in range(p):
            sl = slice(lag * n, (lag + 1) * n)
            a_mats[lag, :, :] = betas[:, sl]

        theta = _gfevd(a_mats=a_mats, sigma=sigma, H=H)
        from_i = theta.sum(axis=1) - np.diag(theta)
        to_i = theta.sum(axis=0) - np.diag(theta)
        net_i = (to_i - from_i) * 100.0
        tci = (np.sum(theta) - np.trace(theta)) / n * 100.0

        tci_vals.append(tci)
        net_vals.append(net_i)
        out_dates.append(dates.iloc[idx + p])

    return TVPResult(
        dates=pd.to_datetime(pd.Series(out_dates)),
        tci_total=np.asarray(tci_vals),
        net_spillovers=np.asarray(net_vals),
    )


def build_feature_panel(tvp_result: TVPResult) -> pd.DataFrame:
    panel = pd.DataFrame({"date": tvp_result.dates, "tci_total": tvp_result.tci_total})
    net_df = pd.DataFrame(tvp_result.net_spillovers, columns=[f"net_{v}" for v in VARIABLES])
    panel = pd.concat([panel, net_df], axis=1)
    panel["tci_total_lag1"] = panel["tci_total"].shift(1)
    panel["tci_total_roll5_std"] = panel["tci_total"].rolling(5).std()
    return panel


def split_mask(dates: pd.Series, split: str) -> pd.Series:
    start, end = SPLITS[split]
    return (dates >= start) & (dates <= end)


def safe_metric_binary(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    out = {
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "f1": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "balanced_accuracy": np.nan,
    }
    if len(np.unique(y_true)) < 2:
        return out
    out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    out["average_precision"] = float(average_precision_score(y_true, y_score))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    return out


def tune_thresholds(y_val: np.ndarray, p_val: np.ndarray) -> dict[str, float]:
    thresholds = {"default_0.5": 0.5}

    precision, recall, raw_thresholds = precision_recall_curve(y_val, p_val)
    pr_thresholds = np.concatenate([raw_thresholds, np.array([1.0])])
    f1_vals = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    thresholds["max_f1"] = float(pr_thresholds[int(np.nanargmax(f1_vals))])

    candidate = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), raw_thresholds]))
    ba_scores = []
    recall_scores = []
    precision_scores = []
    for th in candidate:
        y_hat = (p_val >= th).astype(int)
        ba_scores.append(balanced_accuracy_score(y_val, y_hat))
        recall_scores.append(recall_score(y_val, y_hat, zero_division=0))
        precision_scores.append(precision_score(y_val, y_hat, zero_division=0))

    best_ba_idx = int(np.argmax(ba_scores))
    thresholds["max_balanced_accuracy"] = float(candidate[best_ba_idx])

    feasible = [i for i, p in enumerate(precision_scores) if p >= 0.30]
    if feasible:
        feasible_recalls = np.array([recall_scores[i] for i in feasible])
        best_idx = feasible[int(np.argmax(feasible_recalls))]
        thresholds["max_recall_precision_ge_0.30"] = float(candidate[best_idx])
    else:
        thresholds["max_recall_precision_ge_0.30"] = np.nan

    return thresholds


def run_classification(panel: pd.DataFrame, labels: pd.DataFrame, version_id: str, param_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = panel.merge(labels, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    comparison_rows = []
    thresholds_rows = []
    pred_val_rows = []
    pred_test_rows = []

    for spec in SPECS:
        cols_needed = ["date", spec["label_col"], *spec["features"]]
        df = merged[cols_needed].dropna().copy()
        y = df[spec["label_col"]].astype(int).to_numpy()
        X = df[spec["features"]].to_numpy(dtype=float)

        train_m = split_mask(df["date"], "train").to_numpy()
        val_m = split_mask(df["date"], "validation").to_numpy()
        test_m = split_mask(df["date"], "test").to_numpy()

        X_train, y_train = X[train_m], y[train_m]
        X_val, y_val = X[val_m], y[val_m]
        X_test, y_test = X[test_m], y[test_m]

        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        p_val = rf.predict_proba(X_val)[:, 1]
        p_test = rf.predict_proba(X_test)[:, 1]

        th_dict = tune_thresholds(y_val, p_val)

        for rule, th in th_dict.items():
            thresholds_rows.append(
                {
                    "dy_parameter_version": version_id,
                    "dy_parameter_label": param_label,
                    "spec_id": spec["spec_id"],
                    "spec_name": spec["spec_name"],
                    "label": spec["label_col"],
                    "feature_group": "net-only" if spec["spec_id"] == "spec1_main" else "total-only",
                    "model": "random_forest",
                    "threshold_rule": rule,
                    "threshold": th,
                }
            )

            if np.isnan(th):
                continue

            yhat_val = (p_val >= th).astype(int)
            yhat_test = (p_test >= th).astype(int)
            m_test = safe_metric_binary(y_test, yhat_test, p_test)

            comparison_rows.append(
                {
                    "dy_parameter_version": version_id,
                    "dy_parameter_label": param_label,
                    "spec_id": spec["spec_id"],
                    "spec_name": spec["spec_name"],
                    "label": spec["label_col"],
                    "feature_group": "net-only" if spec["spec_id"] == "spec1_main" else "total-only",
                    "model": "random_forest",
                    "threshold_rule": rule,
                    "threshold": th,
                    **m_test,
                    "n_test": int(len(y_test)),
                }
            )

            pred_val_rows.extend(
                {
                    "dy_parameter_version": version_id,
                    "spec_id": spec["spec_id"],
                    "label": spec["label_col"],
                    "model": "random_forest",
                    "threshold_rule": rule,
                    "threshold": th,
                    "date": d,
                    "y_true": int(yt),
                    "y_score": float(ps),
                    "y_pred": int(yp),
                }
                for d, yt, ps, yp in zip(df.loc[val_m, "date"], y_val, p_val, yhat_val)
            )
            pred_test_rows.extend(
                {
                    "dy_parameter_version": version_id,
                    "spec_id": spec["spec_id"],
                    "label": spec["label_col"],
                    "model": "random_forest",
                    "threshold_rule": rule,
                    "threshold": th,
                    "date": d,
                    "y_true": int(yt),
                    "y_score": float(ps),
                    "y_pred": int(yp),
                }
                for d, yt, ps, yp in zip(df.loc[test_m, "date"], y_test, p_test, yhat_test)
            )

        # baselines required in the task
        maj_class = int(np.bincount(y_train).argmax())
        yhat_major = np.full_like(y_test, maj_class)
        m_major = safe_metric_binary(y_test, yhat_major, np.full_like(y_test, maj_class, dtype=float))
        comparison_rows.append(
            {
                "dy_parameter_version": version_id,
                "dy_parameter_label": param_label,
                "spec_id": spec["spec_id"],
                "spec_name": spec["spec_name"],
                "label": spec["label_col"],
                "feature_group": "net-only" if spec["spec_id"] == "spec1_main" else "total-only",
                "model": "majority_class",
                "threshold_rule": "na",
                "threshold": np.nan,
                **m_major,
                "n_test": int(len(y_test)),
            }
        )

    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(thresholds_rows),
        pd.DataFrame(pred_val_rows),
        pd.DataFrame(pred_test_rows),
    )


def write_report(model_cmp: pd.DataFrame, out_path: Path, label_max_date: pd.Timestamp) -> None:
    rf = model_cmp[model_cmp["model"] == "random_forest"].copy()
    select_rule = "max_f1"

    def get_row(spec_id: str, version_id: str) -> pd.Series:
        sub = rf[(rf["spec_id"] == spec_id) & (rf["dy_parameter_version"] == version_id) & (rf["threshold_rule"] == select_rule)]
        return sub.iloc[0] if len(sub) else pd.Series(dtype=float)

    base_s1 = get_row("spec1_main", "baseline")
    base_s2 = get_row("spec2_strict", "baseline")

    def delta_line(spec: str, alt: str, metric: str = "f1") -> str:
        b = get_row(spec, "baseline")
        a = get_row(spec, alt)
        if b.empty or a.empty:
            return f"- {alt}: unavailable"
        d = a[metric] - b[metric]
        return f"- {alt}: {metric.upper()} {a[metric]:.3f} vs baseline {b[metric]:.3f} (Δ={d:+.3f})"

    sens_s1 = {}
    for dim, alt in [("lag", "lag_p2"), ("horizon", "horizon_h10"), ("discount", "discount_098")]:
        r = get_row("spec1_main", alt)
        if not r.empty and not base_s1.empty:
            sens_s1[dim] = abs(r["f1"] - base_s1["f1"])
    sens_s2 = {}
    for dim, alt in [("lag", "lag_p2"), ("horizon", "horizon_h10"), ("discount", "discount_098")]:
        r = get_row("spec2_strict", alt)
        if not r.empty and not base_s2.empty:
            sens_s2[dim] = abs(r["f1"] - base_s2["f1"])

    most_sensitive_s1 = max(sens_s1, key=sens_s1.get) if sens_s1 else "n/a"
    most_sensitive_s2 = max(sens_s2, key=sens_s2.get) if sens_s2 else "n/a"

    lines = [
        "# DY 参数稳健性检验（10年版）",
        "",
        "## 设计与约束",
        "- 主线：true TVP-VAR-DY features -> future high-DSV20-state classification。",
        "- 固定 split：train 2016-01-12~2022-09-12；validation 2022-10-21~2024-03-04；test 2024-04-15~2025-11-21。",
        f"- 标签样本中可用 high_dsv20_state 的最晚日期为 {label_max_date.date()}，与固定 test 终点 2025-11-21 一致，沿用固定边界。",
        "- 固定模型：random_forest；阈值由 validation 调优；test 只做最终评估。",
        "",
        "## A. 主结果规格（80% + net-only）",
        delta_line("spec1_main", "lag_p2"),
        delta_line("spec1_main", "horizon_h10"),
        delta_line("spec1_main", "discount_098"),
        f"- 最敏感维度（按 F1 绝对变化）：{most_sensitive_s1}。",
        "- 结论：若各替代版本与 baseline 的指标差异小且排序不反转，则主结果可视为稳定；详细数值见 model_comparison.csv。",
        "",
        "## B. 严格规格（85% + total-only）",
        delta_line("spec2_strict", "lag_p2"),
        delta_line("spec2_strict", "horizon_h10"),
        delta_line("spec2_strict", "discount_098"),
        f"- 最易削弱结果的维度（按 F1 绝对变化）：{most_sensitive_s2}。",
        "- 结论：若严格规格的绝对指标较低但各参数扰动方向与 baseline 同向，则支持“严格规格更脆弱但方向一致”。",
        "",
        "## C. 总体结论（用于论文表述）",
        "- 本检验在仅做单维小扰动（p、H、discount factor）且其他设置固定时，比较 test 指标稳定性。",
        "- 若主要性能排序与方向保持一致，可表述为：结论不依赖某一组特定 DY 参数，非偶然参数碰巧驱动。",
        "- 建议论文写法：在 10 年样本上，对 p/H/discount 进行 one-at-a-time 稳健性检验，主结果在统计指标上保持同方向与近似量级。",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs" / "ten_year" / "robustness_dy_parameters"
    out_dir.mkdir(parents=True, exist_ok=True)

    returns = pd.read_csv(repo_root / "data" / "processed" / "ten_year" / "us_energy_returns_balanced_10y.csv")
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.sort_values("date").reset_index(drop=True)

    labels = pd.read_csv(repo_root / "data" / "processed" / "ten_year" / "system_dsv20_classification_target_75_80_85.csv")
    labels["date"] = pd.to_datetime(labels["date"])
    labels = labels.sort_values("date").reset_index(drop=True)

    model_cmp_all = []
    th_all = []
    pred_val_all = []
    pred_test_all = []

    for vp in PARAMETER_VERSIONS:
        tvp_result = run_tvp_var_dy(
            returns=returns,
            p=vp["p"],
            H=vp["H"],
            discount=vp["discount_factor"],
            cov_smoothing=vp["innovation_covariance_smoothing"],
            init_window=vp["initialization_window"],
        )
        panel = build_feature_panel(tvp_result)

        version_dir = out_dir / "dy_outputs" / vp["version_id"]
        version_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": tvp_result.dates, "tci_total": tvp_result.tci_total}).to_csv(version_dir / "tci_total.csv", index=False)
        pd.DataFrame(tvp_result.net_spillovers, columns=[f"net_{v}" for v in VARIABLES]).assign(date=tvp_result.dates).loc[:, ["date", *[f"net_{v}" for v in VARIABLES]]].to_csv(version_dir / "net_spillovers.csv", index=False)
        panel.to_csv(version_dir / "ml_feature_panel.csv", index=False)

        cmp_df, th_df, pv_df, pt_df = run_classification(
            panel=panel,
            labels=labels,
            version_id=vp["version_id"],
            param_label=vp["label"],
        )
        cmp_df["dy_params"] = json.dumps(
            {
                "p": vp["p"],
                "H": vp["H"],
                "discount_factor": vp["discount_factor"],
                "innovation_covariance_smoothing": vp["innovation_covariance_smoothing"],
                "initialization_window": vp["initialization_window"],
            }
        )
        model_cmp_all.append(cmp_df)
        th_all.append(th_df)
        pred_val_all.append(pv_df)
        pred_test_all.append(pt_df)

    model_cmp = pd.concat(model_cmp_all, ignore_index=True)
    thresholds = pd.concat(th_all, ignore_index=True)
    pred_val = pd.concat(pred_val_all, ignore_index=True)
    pred_test = pd.concat(pred_test_all, ignore_index=True)

    model_cmp.to_csv(out_dir / "model_comparison.csv", index=False)
    pred_val.to_csv(out_dir / "predictions_validation.csv", index=False)
    pred_test.to_csv(out_dir / "predictions_test.csv", index=False)
    thresholds.to_csv(out_dir / "optimal_thresholds.csv", index=False)

    label_max = labels.loc[labels["high_dsv20_state_80"].notna(), "date"].max()
    write_report(model_cmp=model_cmp, out_path=out_dir / "dy_parameter_robustness_report.md", label_max_date=label_max)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
