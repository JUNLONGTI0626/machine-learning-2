# SKILL: prepare-analysis-data

## Purpose
Convert aligned daily level data into analysis-ready return data.

## Use this skill when
- Level data have already been aligned
- Returns are needed for connectedness estimation
- Basic descriptive checks are required

## Inputs
- `data/interim/us_energy_levels_aligned.csv`

## Outputs
- `data/processed/us_energy_returns.csv`
- `outputs/reports/analysis_data_report.md`

## Steps
1. Sort by date
2. Compute log returns for all variables
3. Drop the first row created by differencing
4. Check missing values and extreme outliers
5. Save analysis-ready return panel

## Validation checks
- Returns are finite
- Date sequence is preserved
- No accidental forward-looking transformation is used
