# U.S. Energy Spillover and ML Early Warning

This repository supports research on the U.S. energy market using time-varying connectedness models and machine-learning-based early warning.

## Research objective

The project focuses on three linked tasks:

1. build a clean daily dataset for major U.S. energy-market variables;
2. estimate time-frequency spillovers using the TVP-VAR-BK framework;
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

1. ingest raw level data;
2. align dates and clean formats;
3. compute log returns;
4. estimate TVP-VAR-BK spillovers;
5. export total, short-term, and long-term connectedness;
6. construct machine-learning features from spillover outputs;
7. construct a single forward-looking output variable based on system Expected Shortfall;
8. evaluate ML readiness before formal modeling.

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
│       ├── run-tvpvar-bk/
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
