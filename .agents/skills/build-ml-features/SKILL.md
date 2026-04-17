# SKILL: build-ml-features

## Purpose
Transform TVP-VAR-BK outputs into machine-learning feature matrices.

## Use this skill when
- Spillover outputs have already been estimated
- ML-ready inputs are needed
- Feature engineering for forecasting is required

## Inputs
- TVP-VAR-BK output tables from `outputs/tables/`

## Outputs
- `data/processed/ml_feature_panel.csv`
- `outputs/reports/ml_feature_build_report.md`

## Steps
1. Load total, short, and long connectedness series
2. Add directional spillovers: TO, FROM, NET
3. Create lagged features
4. Create rolling means and rolling standard deviations
5. Align all features by date
6. Save ML feature panel

## Validation checks
- All features use only time-t information
- No forward-looking leakage
- Features have sufficient variation
- Strong redundancy is documented

## Common failure modes
- Features accidentally use t+1 information
- Too many highly collinear variables
- Large sample loss after lagging
