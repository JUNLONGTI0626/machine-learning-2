# SKILL: evaluate-ml-readiness

## Purpose
Assess whether TVP-VAR-BK outputs are suitable as machine-learning inputs.

## Use this skill when
- Spillover outputs have been generated
- Before launching ML modeling
- To decide whether features carry enough information

## Inputs
- `data/processed/ml_feature_panel.csv`
- `data/processed/system_es_target.csv`

## Outputs
- `outputs/reports/ml_readiness_report.md`

## Steps
1. Check sample size
2. Check missing values
3. Check feature variation
4. Check correlations across features
5. Check date alignment with target
6. Assess leakage risk
7. Conclude whether ML is feasible

## Validation checks
- X_t and Y_{t+1} are correctly aligned
- Features are not nearly constant
- Sample length is adequate
- Leakage risk is explicitly discussed
