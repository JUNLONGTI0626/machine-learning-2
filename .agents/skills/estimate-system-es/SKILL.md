# SKILL: estimate-system-es

## Purpose
Construct the single forward-looking output variable based on system Expected Shortfall.

## Use this skill when
- Return data are ready
- The ML target variable needs to be constructed
- A single paper-level output variable is required

## Inputs
- `data/processed/us_energy_returns.csv`

## Outputs
- `data/processed/system_es_target.csv`
- `outputs/reports/system_es_report.md`

## Steps
1. Construct a system return series from the selected energy variables
2. Use a transparent weighting scheme
3. Estimate forward-looking Expected Shortfall
4. Align target dates with feature dates
5. Save target variable

## Validation checks
- Weighting scheme is documented
- ES definition is explicit
- Target is forward-looking rather than contemporaneous
- Target is aligned correctly with X_t

## Common failure modes
- Mixing level data with returns
- Using contemporaneous ES instead of future ES
- Misaligned target dates
