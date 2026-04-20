# Data Cleaning and Merge Report

## 1) Files scanned and variable identification

- **brent_daily.csv**
  - mapped variable: `brent`
  - detected date column: `observation_date`
  - selected value column: `DCOILBRENTEU`
  - numeric candidates: `DCOILBRENTEU`
  - status: included
- **cels_daily.csv**
  - mapped variable: `cels`
  - detected date column: `Date`
  - selected value column: `Close/Last`
  - numeric candidates: `Close/Last`, `Open`, `High`, `Low`
  - status: included
- **henry_hub_daily.csv**
  - mapped variable: `henry_hub`
  - detected date column: `observation_date`
  - selected value column: `DHHNGSP`
  - numeric candidates: `DHHNGSP`
  - status: included
- **pjm_west_daily.xlsx**
  - mapped variable: `pjm_west`
  - detected date column: `Trade date`
  - selected value column: `Wtd avg price $/MWh`
  - numeric candidates: `High price $/MWh`, `Low price $/MWh`, `Wtd avg price $/MWh`, `Change`, `Daily volume MWh`, `Number of trades`, `Number of counterparties`
  - status: included
- **rbob_daily.csv**
  - mapped variable: `rbob`
  - detected date column: `observation_date`
  - selected value column: `DRGASLA`
  - numeric candidates: `DRGASLA`
  - status: included
- **wti_daily.csv**
  - mapped variable: `wti`
  - detected date column: `observation_date`
  - selected value column: `DCOILWTICO`
  - numeric candidates: `DCOILWTICO`
  - status: included

## 2) Original sample ranges by variable

| variable | file | start_date | end_date | raw_rows | clean_rows |
|---|---|---:|---:|---:|---:|
| brent | brent_daily.csv | 2021-01-04 | 2025-12-31 | 1303 | 1303 |
| cels | cels_daily.csv | 2021-01-04 | 2025-12-31 | 1255 | 1255 |
| henry_hub | henry_hub_daily.csv | 2021-01-04 | 2025-12-31 | 1303 | 1303 |
| pjm_west | pjm_west_daily.xlsx | 2021-01-04 | 2025-12-31 | 1196 | 1195 |
| rbob | rbob_daily.csv | 2021-01-04 | 2025-12-31 | 1303 | 1303 |
| wti | wti_daily.csv | 2021-01-04 | 2025-12-31 | 1303 | 1303 |

## 3) Cleaning actions taken

- **brent** (brent_daily.csv): Text/symbol noise removed: 40 rows
- **cels** (cels_daily.csv): Date order normalized to ascending
- **henry_hub** (henry_hub_daily.csv): Text/symbol noise removed: 54 rows
- **pjm_west** (pjm_west_daily.xlsx): Duplicate dates dropped (keep last): 1
- **rbob** (rbob_daily.csv): Text/symbol noise removed: 56 rows
- **wti** (wti_daily.csv): Text/symbol noise removed: 55 rows

## 5) Missingness after outer merge

| variable | missing_count | missing_ratio |
|---|---:|---:|
| pjm_west | 108 | 8.29% |
| rbob | 56 | 4.30% |
| wti | 55 | 4.22% |
| henry_hub | 54 | 4.14% |
| cels | 48 | 3.68% |
| brent | 40 | 3.07% |

## 6) Final aligned sample

- sample start date: **2021-01-04**
- sample end date: **2025-12-31**
- total observations (daily index rows): **1303**
