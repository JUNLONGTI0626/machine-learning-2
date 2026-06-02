#!/usr/bin/env python3
"""Build 10-year forward DSV20 classification targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("data/processed/ten_year/us_energy_returns_balanced_10y.csv")
OUTPUT_PATH = Path("data/processed/ten_year/system_dsv20_classification_target_10y.csv")
MARKET_COLS = ["wti", "henry_hub", "brent", "rbob", "pjm_west", "cels"]
FORWARD_HORIZON = 20


def build_forward_dsv20(system_return: pd.Series, horizon: int = FORWARD_HORIZON) -> pd.Series:
    """Compute forward 20-day downside semivariance-style risk measure."""
    neg_sq = np.square(np.minimum(system_return, 0.0))
    forward_terms = [neg_sq.shift(-h) for h in range(1, horizon + 1)]
    return pd.concat(forward_terms, axis=1).sum(axis=1, min_count=horizon)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    required_cols = {"date", *MARKET_COLS}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["system_return"] = df[MARKET_COLS].mean(axis=1)
    df["system_dsv_forward_20d"] = build_forward_dsv20(df["system_return"], FORWARD_HORIZON)

    history_dsv = df["system_dsv_forward_20d"].shift(1)
    df["rolling_threshold_80"] = history_dsv.expanding(min_periods=1).quantile(0.80)
    df["rolling_threshold_85"] = history_dsv.expanding(min_periods=1).quantile(0.85)

    df["high_dsv20_state_80"] = (
        df["system_dsv_forward_20d"] > df["rolling_threshold_80"]
    ).astype("float")
    df["high_dsv20_state_85"] = (
        df["system_dsv_forward_20d"] > df["rolling_threshold_85"]
    ).astype("float")

    out_cols = [
        "date",
        "system_return",
        "system_dsv_forward_20d",
        "rolling_threshold_80",
        "rolling_threshold_85",
        "high_dsv20_state_80",
        "high_dsv20_state_85",
    ]
    out = df[out_cols].dropna().copy()
    out[["high_dsv20_state_80", "high_dsv20_state_85"]] = out[
        ["high_dsv20_state_80", "high_dsv20_state_85"]
    ].astype(int)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(out):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
