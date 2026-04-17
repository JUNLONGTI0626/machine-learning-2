# SKILL: ingest-energy-data

## Purpose
Ingest raw U.S. energy-market data files and produce a single aligned daily level dataset.

## Use this skill when
- New raw data files have been uploaded
- Variable names or date formats are inconsistent
- A unified daily panel is needed before return construction

## Inputs
- Raw CSV/XLSX files in `data/raw/`
- Expected variables:
  - wti
  - henry_hub
  - brent
  - rbob
  - pjm_west or ercot_north
  - cels

## Outputs
- `data/interim/us_energy_levels_aligned.csv`
- `outputs/reports/data_ingestion_report.md`

## Steps
1. Load all raw files
2. Standardize date format
3. Rename variables to lowercase snake_case
4. Merge on date
5. Preserve missing values and document them
6. Save aligned level dataset

## Validation checks
- Date column exists and is unique
- Variables are numeric
- Merge does not create duplicated dates
- Missing-value rate is reported

## Common failure modes
- Mixed date formats
- Thousand separators or text symbols in numeric columns
- Non-overlapping date ranges
