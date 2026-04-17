# SKILL: run-tvpvar-dy

## Purpose
Estimate dynamic spillovers using the TVP-VAR-DY framework for the U.S. energy market.

## Use this skill when
- analysis-ready balanced returns are available
- time-varying connectedness needs to be estimated
- ML features will be built from dynamic spillover outputs

## Inputs
- data/processed/us_energy_returns_balanced.csv

## Outputs
- outputs/tvpvar_dy_baseline/tci_total.csv
- outputs/tvpvar_dy_baseline/directional_to.csv
- outputs/tvpvar_dy_baseline/directional_from.csv
- outputs/tvpvar_dy_baseline/net_spillovers.csv
- outputs/tvpvar_dy_baseline/tci_total.png
- outputs/tvpvar_dy_baseline/tvpvar_dy_baseline_report.md

## Steps
1. load balanced return data
2. estimate TVP-VAR-DY connectedness
3. export total connectedness
4. export TO, FROM, NET spillovers for all variables
5. plot the dynamic total connectedness
6. write a concise report on output quality and ML readiness

## Validation checks
- total connectedness is generated successfully
- TO/FROM/NET series are aligned on the same dates
- outputs are not constant or near-constant
- dates are preserved correctly

## Common failure modes
- insufficient balanced sample length
- unstable parameters
- missing directional spillover exports
