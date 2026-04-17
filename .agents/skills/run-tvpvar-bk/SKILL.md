# SKILL: run-tvpvar-bk

## Purpose
Estimate time-frequency spillovers using the TVP-VAR-BK framework.

## Use this skill when
- Analysis-ready returns are available
- Dynamic connectedness estimation is required
- Spillover outputs are needed for plots, tables, or ML feature construction

## Inputs
- `data/processed/us_energy_returns.csv`

## Outputs
- `outputs/tables/tvpvar_bk_total_connectedness.csv`
- `outputs/tables/tvpvar_bk_short_connectedness.csv`
- `outputs/tables/tvpvar_bk_long_connectedness.csv`
- `outputs/tables/tvpvar_bk_directional_spillovers.csv`
- `outputs/figures/tvpvar_bk_total.png`
- `outputs/figures/tvpvar_bk_short.png`
- `outputs/figures/tvpvar_bk_long.png`
- `outputs/reports/tvpvar_bk_run_report.md`

## Steps
1. Load return data
2. Select lag order and forecast horizon
3. Run TVP-VAR-BK
4. Export total, short, and long connectedness
5. Export TO, FROM, and NET spillovers
6. Plot key time-series outputs
7. Write a concise run report

## Validation checks
- Total, short, and long spillovers are successfully generated
- Outputs are not constant or near-constant
- Dates align with the original return sample
- Frequency components are interpretable

## Common failure modes
- Insufficient sample size
- Parameter instability
- Missing results from one or more frequency bands
