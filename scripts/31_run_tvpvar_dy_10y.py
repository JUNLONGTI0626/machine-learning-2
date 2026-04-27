#!/usr/bin/env python3
"""Run true TVP-VAR-DY on the ten-year balanced U.S. energy returns sample."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


INPUT_PATH = Path("data/processed/ten_year/us_energy_returns_balanced_10y.csv")
OUTPUT_DIR = Path("outputs/ten_year/tvpvar_dy_baseline")
REPORT_PATH = OUTPUT_DIR / "tvpvar_dy_baseline_report_10y.md"

VARIABLES = ["wti", "henry_hub", "brent", "rbob", "pjm_west", "cels"]
MAX_AUTO_LAG = 5

# Baseline hyper-parameters requested by the task.
HORIZON = 20
DISCOUNT_FACTOR = 0.99
COV_SMOOTHING = 0.96
INIT_WINDOW = 250


@dataclass
class TVPResults:
    dates: pd.Series
    tci_total: pd.Series
    directional_to: pd.DataFrame
    directional_from: pd.DataFrame
    net_spillovers: pd.DataFrame
    lag_order: int


def load_returns(path: Path, variables: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Input data must include a 'date' column.")

    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"Missing required variables: {missing}")

    out = df[["date", *variables]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(subset="date", keep="last")
    out = out.dropna(subset=variables).reset_index(drop=True)
    return out


def choose_lag_order(data: pd.DataFrame, maxlags: int = MAX_AUTO_LAG) -> Tuple[int, str]:
    try:
        fitted = VAR(data).select_order(maxlags=maxlags)
        p = int(fitted.aic)
        if p < 1:
            p = 1
        return p, "AIC automatic selection via statsmodels VAR.select_order"
    except Exception as exc:  # pragma: no cover - safety branch
        return 2, f"Automatic lag selection failed ({exc}); fallback to p=2"


def make_regression_row(y_hist: np.ndarray, p: int, with_intercept: bool = True) -> np.ndarray:
    # y_hist has shape (p, n), with most recent value at y_hist[-1].
    lags = [y_hist[-lag].ravel() for lag in range(1, p + 1)]
    x = np.concatenate(lags)
    if with_intercept:
        x = np.concatenate([np.array([1.0]), x])
    return x


def companion_matrix(beta_t: np.ndarray, n_vars: int, p: int) -> np.ndarray:
    # beta_t is (n_vars, k) where k = 1 + n_vars * p (intercept + lag coefficients)
    coeff = beta_t[:, 1:]  # remove intercept for FEVD dynamics
    a_blocks = [coeff[:, i * n_vars : (i + 1) * n_vars] for i in range(p)]

    dim = n_vars * p
    f = np.zeros((dim, dim))
    f[:n_vars, : n_vars * p] = np.concatenate(a_blocks, axis=1)
    if p > 1:
        f[n_vars:, :-n_vars] = np.eye(n_vars * (p - 1))
    return f


def generalized_fevd(beta_t: np.ndarray, sigma_t: np.ndarray, n_vars: int, p: int, horizon: int) -> np.ndarray:
    f = companion_matrix(beta_t, n_vars=n_vars, p=p)
    j = np.hstack([np.eye(n_vars), np.zeros((n_vars, n_vars * (p - 1)))])

    sigma = sigma_t.copy()
    # Stabilize covariance matrix for numerical safety.
    sigma = (sigma + sigma.T) / 2.0
    eigvals = np.linalg.eigvalsh(sigma)
    min_eig = np.min(eigvals)
    if min_eig <= 1e-10:
        sigma += np.eye(n_vars) * (1e-10 - min_eig + 1e-10)

    numer = np.zeros((n_vars, n_vars))
    denom = np.zeros(n_vars)

    f_power = np.eye(n_vars * p)
    for _ in range(horizon):
        psi_h = j @ f_power @ j.T
        tmp = psi_h @ sigma
        for i in range(n_vars):
            denom[i] += psi_h[i, :] @ sigma @ psi_h[i, :].T
            for j_idx in range(n_vars):
                numer[i, j_idx] += (tmp[i, j_idx] ** 2) / max(sigma[j_idx, j_idx], 1e-12)
        f_power = f_power @ f

    fevd = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        if denom[i] <= 1e-12:
            continue
        fevd[i, :] = numer[i, :] / denom[i]
        row_sum = fevd[i, :].sum()
        if row_sum > 1e-12:
            fevd[i, :] /= row_sum
    return fevd


def run_true_tvp_var_dy(
    returns: pd.DataFrame,
    variables: list[str],
    p: int,
    horizon: int,
    discount_factor: float,
    cov_smoothing: float,
    init_window: int,
) -> TVPResults:
    y = returns[variables].to_numpy(dtype=float)
    dates = returns["date"]
    t_total, n_vars = y.shape

    k = 1 + n_vars * p
    # State per equation: coefficients follow random walk.
    beta = np.zeros((n_vars, k))
    p_state = np.array([np.eye(k) * 0.1 for _ in range(n_vars)])

    # Initial covariance from first init_window observations.
    init_slice = y[: max(init_window, p + 5), :]
    sigma = np.cov(init_slice.T)
    sigma = np.atleast_2d(sigma)

    records_tci: list[float] = []
    records_to: list[np.ndarray] = []
    records_from: list[np.ndarray] = []
    records_net: list[np.ndarray] = []
    record_dates: list[pd.Timestamp] = []

    for t in range(p, t_total):
        x_t = make_regression_row(y[t - p : t, :], p=p, with_intercept=True)

        residuals = np.zeros(n_vars)
        for i in range(n_vars):
            beta_pred = beta[i].copy()
            p_pred = p_state[i] / discount_factor

            y_hat = x_t @ beta_pred
            resid = y[t, i] - y_hat
            residuals[i] = resid

            s_t = x_t @ p_pred @ x_t.T + max(sigma[i, i], 1e-10)
            k_gain = (p_pred @ x_t) / s_t

            beta_upd = beta_pred + k_gain * resid
            p_upd = p_pred - np.outer(k_gain, x_t) @ p_pred
            p_upd = (p_upd + p_upd.T) / 2.0

            beta[i] = beta_upd
            p_state[i] = p_upd

        sigma = cov_smoothing * sigma + (1.0 - cov_smoothing) * np.outer(residuals, residuals)
        sigma = (sigma + sigma.T) / 2.0

        fevd_t = generalized_fevd(beta_t=beta, sigma_t=sigma, n_vars=n_vars, p=p, horizon=horizon)

        off_diag_sum = fevd_t.sum() - np.trace(fevd_t)
        tci_t = 100.0 * off_diag_sum / n_vars

        to_i = 100.0 * (fevd_t.sum(axis=0) - np.diag(fevd_t)) / n_vars
        from_i = 100.0 * (fevd_t.sum(axis=1) - np.diag(fevd_t)) / n_vars
        net_i = to_i - from_i

        records_tci.append(float(tci_t))
        records_to.append(to_i)
        records_from.append(from_i)
        records_net.append(net_i)
        record_dates.append(dates.iloc[t])

    idx_dates = pd.Series(record_dates, name="date")
    to_df = pd.DataFrame(records_to, columns=[f"to_{v}" for v in variables])
    from_df = pd.DataFrame(records_from, columns=[f"from_{v}" for v in variables])
    net_df = pd.DataFrame(records_net, columns=[f"net_{v}" for v in variables])

    return TVPResults(
        dates=idx_dates,
        tci_total=pd.Series(records_tci, name="tci_total"),
        directional_to=to_df,
        directional_from=from_df,
        net_spillovers=net_df,
        lag_order=p,
    )


def detect_anomalies(series: pd.Series) -> Dict[str, bool]:
    std = float(series.std(ddof=0))
    value_range = float(series.max() - series.min())
    near_constant = std < 1e-3 or value_range < 1e-2

    diff_std = float(series.diff().dropna().std(ddof=0)) if len(series) > 2 else 0.0
    overly_smooth = diff_std < 1e-3

    invalid = bool((~np.isfinite(series)).any() or (series.abs() > 1e6).any())
    return {
        "near_constant": near_constant,
        "overly_smooth": overly_smooth,
        "invalid_values": invalid,
    }


def save_outputs(results: TVPResults, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    tci_df = pd.DataFrame({"date": results.dates, "tci_total": results.tci_total})
    to_df = pd.concat([results.dates, results.directional_to], axis=1)
    from_df = pd.concat([results.dates, results.directional_from], axis=1)
    net_df = pd.concat([results.dates, results.net_spillovers], axis=1)

    tci_df.to_csv(out_dir / "tci_total.csv", index=False)
    to_df.to_csv(out_dir / "directional_to.csv", index=False)
    from_df.to_csv(out_dir / "directional_from.csv", index=False)
    net_df.to_csv(out_dir / "net_spillovers.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(tci_df["date"], tci_df["tci_total"], lw=1.4)
    plt.title("True TVP-VAR-DY Total Connectedness Index (10-year sample)")
    plt.xlabel("Date")
    plt.ylabel("TCI")
    plt.tight_layout()
    plt.savefig(out_dir / "tci_total.png", dpi=150)
    plt.close()


def build_report(
    returns: pd.DataFrame,
    results: TVPResults,
    lag_msg: str,
    out_path: Path,
    horizon: int,
    discount_factor: float,
    cov_smoothing: float,
    init_window: int,
) -> None:
    tci = results.tci_total
    anomaly = detect_anomalies(tci)

    net_means = results.net_spillovers.mean().sort_values(ascending=False)
    net_rank_lines = "\n".join([f"{i+1}. {name.replace('net_', '')}: {val:.4f}" for i, (name, val) in enumerate(net_means.items())])

    suitability = "适合"
    if anomaly["near_constant"] or anomaly["invalid_values"]:
        suitability = "需谨慎"

    content = f"""# 10年样本 True TVP-VAR-DY 基线报告

## 1) 实现性质
- 本次实现为**真正的 TVP-VAR-DY**：系数向量按时点递归更新，不使用 rolling window，也未采用“窗口重估 VAR”的方式。
- 更新机制：对每个方程使用 Kalman 递归（state random walk），并结合折扣因子进行状态协方差递推；残差协方差采用指数平滑更新。

## 2) 关键参数
- 滞后阶数 p = {results.lag_order}（{lag_msg}）
- FEVD horizon H = {horizon}
- discount factor = {discount_factor}
- innovation covariance smoothing = {cov_smoothing}
- initialization window = {init_window}

## 3) 样本区间与有效输出区间
- 原始样本起止：{returns['date'].min().date()} 至 {returns['date'].max().date()}
- 有效 TVP 输出：{results.dates.min().date()} 至 {results.dates.max().date()}
- 有效输出样本量：{len(results.dates)}

## 4) TCI 描述统计
- 均值：{tci.mean():.6f}
- 标准差：{tci.std(ddof=0):.6f}
- 最小值：{tci.min():.6f}
- 最大值：{tci.max():.6f}

## 5) 各市场平均净溢出（降序）
{net_rank_lines}

## 6) 数值与平滑性检查
- 异常平滑（差分波动极低）：{anomaly['overly_smooth']}
- 几乎不波动（近常数）：{anomaly['near_constant']}
- 数值异常（NaN/Inf/极端值）：{anomaly['invalid_values']}

## 7) 对后续10年版分类主线特征可用性判断
- 结论：当前 DY 输出**{suitability}**作为后续分类主线输入特征。
- 说明：若后续模型阶段发现边际贡献偏低，可在不改变 true TVP 框架前提下，对 discount / smoothing 做稳健性比较。
"""
    out_path.write_text(content, encoding="utf-8")


def main() -> None:
    returns = load_returns(INPUT_PATH, VARIABLES)
    p, lag_msg = choose_lag_order(returns[VARIABLES], maxlags=MAX_AUTO_LAG)

    results = run_true_tvp_var_dy(
        returns=returns,
        variables=VARIABLES,
        p=p,
        horizon=HORIZON,
        discount_factor=DISCOUNT_FACTOR,
        cov_smoothing=COV_SMOOTHING,
        init_window=INIT_WINDOW,
    )

    save_outputs(results, OUTPUT_DIR)
    build_report(
        returns=returns,
        results=results,
        lag_msg=lag_msg,
        out_path=REPORT_PATH,
        horizon=HORIZON,
        discount_factor=DISCOUNT_FACTOR,
        cov_smoothing=COV_SMOOTHING,
        init_window=INIT_WINDOW,
    )

    print(f"Saved TVP-VAR-DY outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
