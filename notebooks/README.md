# Notebooks

## Overview

The notebooks directory is a collaborative workspace for exploratory analysis, practical prototypes, tutorials, and reproducible research related to the Real-Time Market Intelligence Platform. It helps align data science discovery with product development while keeping experiments organized and shareable.

## Purpose

- Exploratory data analysis (EDA) and hypothesis validation
- Prototyping of features, models, and data pipelines
- Step-by-step tutorials for internal and external users
- Benchmarking experiments and performance investigations
- Reproducible research with clear environment and data handling

## Directory Structure

```
notebooks/
├── eda/                 # Exploratory analysis
│   └── 2025-09-29-market-eda.ipynb
├── tutorials/           # Guided walkthroughs and how-tos
│   ├── getting-started.ipynb
│   └── api-client-usage.ipynb
├── prototypes/          # Model and feature prototypes
│   ├── sentiment-prototype.ipynb
│   └── forecasting-lstm-prototype.ipynb
├── benchmarks/          # Performance and accuracy benchmarks
│   └── clickhouse-query-benchmarks.ipynb
├── utils/               # Shared helpers for notebooks
│   ├── plotting.py
│   └── data_loading.py
├── data/                # Local sample data (gitignored)
├── templates/           # Notebook templates
│   ├── eda_template.ipynb
│   └── tutorial_template.ipynb
└── README.md            # This file
```

Note: notebooks/data/ should be excluded via .gitignore to avoid committing large or sensitive data. Use small, sanitized samples when needed.

## Environment

Recommended options to ensure reproducibility:

- Python 3.9+ with virtualenv or conda
- JupyterLab >= 4 or VSCode notebooks
- Pin dependencies in requirements-dev.txt
- Use environment variables from .env for secrets (never hardcode)
- Optionally use ipykernel to register per-project kernels

Setup example:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
python -m ipykernel install --user --name rtm --display-name "RTM Notebook"
```

## Data Handling

- Do not store raw production data in this repo
- Use synthetic or anonymized samples
- Load data securely via environment variables and read-only credentials
- Document data sources and sampling procedures in each notebook
- Cache transient data in local tmp folders (gitignored)

## Naming Conventions

- Use kebab-case for files: `market-eda-aapl-2025-09.ipynb`
- Prefix with date for EDA logs: `YYYY-MM-DD-<topic>.ipynb`
- Keep titles and top-level markdown cells descriptive

## Template: EDA

First cell (metadata):
```
# Title: Market EDA for AAPL
# Author: <name>
# Date: 2025-09-29
# Objective: Explore intraday price behavior and volume anomalies
```

Core sections:
- Context & questions
- Data loading and sampling
- Quality checks (nulls, ranges, duplicates)
- Exploratory plots and summary stats
- Findings and next steps

## Template: Tutorial

Outline:
- Overview and prerequisites
- Setup (env vars, API keys)
- Step-by-step walkthrough with code cells
- Validation/expected outputs
- Troubleshooting and tips

## Good Practices

- Keep notebooks idempotent; top-to-bottom execution should succeed
- Limit notebook runtime (<10 min) and data size
- Parameterize where possible (papermill or environment variables)
- Extract reusable code to notebooks/utils and import it
- Use clear headings and comments to guide readers

## Versioning & Review

- Commit only meaningful checkpoints; clear out temporary cells/outputs
- Clear outputs before committing unless they add high value
- Code review for prototypes prior to promotion into src/

## Useful Commands

```
# Start JupyterLab
jupyter lab

# Convert to HTML/PDF (optional)
jupyter nbconvert --to html notebooks/eda/2025-09-29-market-eda.ipynb

# Run parameterized notebook with papermill (optional)
papermill input.ipynb output.ipynb -p symbol AAPL -p days 30
```

## Next Steps

- Add starter notebooks in eda/, tutorials/, prototypes/
- Create small sanitized datasets in notebooks/data/ (gitignored)
- Register ipykernel and document dependencies
- Promote validated code to src/ with tests
