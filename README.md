# U.S. Energy Spillover and ML Early Warning

This repository supports research on the U.S. energy market using time-varying connectedness models and machine-learning-based early warning.

## Research objective

The project focuses on three linked tasks:

1. build a clean daily dataset for major U.S. energy-market variables;
2. estimate time-frequency spillovers using the TVP-VAR-DY framework;
3. evaluate whether spillover outputs can be used as machine-learning features to predict a forward-looking system Expected Shortfall.

## Core variables

The baseline U.S. energy-market system currently includes:

- WTI crude oil
- Henry Hub natural gas
- Brent crude oil
- RBOB gasoline
- U.S. wholesale electricity price (e.g., PJM West or ERCOT North)
- Nasdaq Clean Edge Green Energy Index (CELS)

## Main workflow

The empirical workflow follows these steps:

clean raw data
compute returns
run TVP-VAR-DY
export TCI / TO / FROM / NET
build DY-based ML features
merge with forward 60d system ES
run baseline ML

## Repository structure

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .agents/
│   ├── registry.md
│   └── skills/
│       ├── ingest-energy-data/
│       ├── prepare-analysis-data/
│       ├── run-tvpvar-DY/
│       ├── build-ml-features/
│       ├── estimate-system-es/
│       ├── evaluate-ml-readiness/
│       └── draft-paper/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── logs/
│   └── reports/
├── scripts/
└── src/
