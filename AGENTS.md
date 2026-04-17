## Project mission
This repository supports research on the U.S. energy market using time-varying connectedness models and machine-learning-based early warning.

Current project goal:
1. build a clean daily dataset for the U.S. energy market;
2. estimate dynamic connectedness using TVP-VAR-DY;
3. construct ML features from total and directional spillovers;
4. predict forward 60-day system Expected Shortfall.

## Working principles
- Prefer reproducible scripts over manual operations.
- Save all intermediate outputs to structured folders.
- Never overwrite key outputs silently.
- If a file format problem appears, fix it in code and document it in the report.
- Do not skip errors; explain where and why they occurred.
- Keep naming consistent across scripts, tables, and figures.

## Data conventions
- Raw data go to `data/raw/`
- Cleaned aligned level data go to `data/interim/`
- Analysis-ready return data go to `data/processed/`
- Date column must be named `date`
- Variable names must be lowercase with underscores

## Expected core variables
- wti
- henry_hub
- brent
- rbob
- pjm_west (or ercot_north)
- cels

## Output conventions
- Tables: `outputs/tables/`
- Figures: `outputs/figures/`
- Logs: `outputs/logs/`
- Reports: `outputs/reports/`

## Modeling conventions
- Use log returns unless otherwise specified.
- TVP-VAR-BK outputs should include total, short-term, and long-term connectedness.
- Machine-learning inputs must be constructed using only information available at time t.
- The single output variable for ML is forward-looking system Expected Shortfall.

## Writing conventions
- Writing style should be concise and journal-oriented.
- Avoid inflated claims and overly causal language.
- Keep the tone close to Energy Economics.
- Do not use AI-like transition phrases or repetitive stock wording.

## Skill routing
When a task arrives, prefer the following skill order:
1. ingest-energy-data
2. prepare-analysis-data
3. run-tvpvar-bk
4. build-ml-features
5. estimate-system-es
6. evaluate-ml-readiness
7. draft-paper
