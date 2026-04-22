# Drug Policy Criminalization — Experiment Guide

## What This Project Is

This project asks: **which US states treat drug use as a criminal problem vs. a public health problem, and what explains the difference?**

It measures that by computing a **Criminalization Index** for each state, each year from 2015–2022:

```
Criminalization Index = drug arrest rate per 100k / drug treatment admission rate per 100k
```

- Index **above 1.0** → state arrests more people than it treats (criminalization-oriented)
- Index **below 1.0** → state treats more people than it arrests (treatment-oriented)
- National median: **0.996** (nearly balanced)
- Most criminalizing state: **Idaho** (average index 6.66)
- Most treatment-oriented: **Connecticut** (average index 0.13)

Machine learning models are then trained on demographic, political, and structural features to explain why some states score 50× higher than others.

---

## Architecture Overview

Data flows in one direction — raw sources → merged dataset → model → outputs:

```
External APIs / CSVs
        ↓
  [fetch_*.py / parse_*.py]   pull and clean each data source independently
        ↓
  [02_data_processing.py]     merge everything; compute Criminalization Index
        ↓
  data/processed/panel_dataset.csv   ← single file everything downstream reads
        ↓
  [03_eda.py]                 explore and summarize
  [04_modeling.py]            train and evaluate models
  [05_visualizations.py]      produce final charts
        ↓
  outputs/                    HTML charts, PNGs, text summaries
```

There is no database, no web server, no live pipeline. Everything runs locally as Python scripts. The notebook re-implements the same pipeline interactively so you can tweak settings and re-run.

---

## File and Module Responsibilities

### Data Fetching / Parsing (run once to build raw data)

These only need to run if the raw CSVs in `data/raw/` are missing or you want to refresh data.

| File | What it does | Output |
|---|---|---|
| `src/fetch_acs.py` | Calls Census Bureau API; downloads population, poverty, race, income for all states 2015–2022 | `data/raw/acs/acs_2015_2022_combined.csv` |
| `src/fetch_nibrs.py` | Calls FBI Crime Data Explorer API; downloads drug arrest counts per state per year | `data/raw/nibrs/nibrs_2015_2022_combined.csv` |
| `src/fetch_cdc_overdose.py` | Calls CDC API; downloads drug overdose death counts | `data/raw/cdc/cdc_overdose_2015_2022.csv` |
| `src/fetch_legislature.py` | Builds state legislature partisan seat percentages | `data/raw/policy/legislature_control.csv` |
| `src/fetch_policy_features.py` | Creates flags for recreational marijuana legalization and governor party | `data/raw/policy/` |
| `src/fetch_political_features.py` | Computes presidential vote share and governor party streak per state-year | `data/raw/policy/political_features.csv` |
| `src/parse_nibrs.py` | Parses raw FBI arrest files (fixed-width text 2015–2019, Excel 2020–2022) | same as fetch_nibrs output |
| `src/parse_teds.py` | Parses the SAMHSA treatment admissions dataset (~1M rows) | `data/raw/teds/teds_2015_2022_combined.csv` |
| `src/parse_bjs.py` | Parses Vera Institute / BJS incarceration data | `data/raw/bjs/bjs_incarceration_2015_2022.csv` |
| `src/parse_lee.py` | Parses FBI law enforcement employees (police per capita) — **not used in final model** | `data/raw/lee/lee_state_2015_2022.csv` |

### Core Pipeline Scripts

| File | What it does |
|---|---|
| `src/02_data_processing.py` | **Most important.** Merges all sources, removes bad data (Florida 2017–2021, Oregon), computes the Criminalization Index, writes `panel_dataset.csv` |
| `src/03_eda.py` | Loads panel dataset, produces exploratory charts and `eda_summary.txt` |
| `src/04_modeling.py` | Trains 5 models with cross-validation, runs hyperparameter tuning, writes `model_results.txt` |
| `src/05_visualizations.py` | Produces 5 final polished Plotly charts |
| `src/01_data_acquisition.py` | Empty stub — not implemented |

### Key Data File

`data/processed/panel_dataset.csv` — **377 rows** (state-year observations), **49 states** (Oregon excluded), **2015–2022**. Every downstream script reads only this file. Never edit it by hand — regenerate it by re-running `02_data_processing.py`.

---

## Data Flow

Here is how one data point — Idaho's 2019 Criminalization Index of 14.1 — gets created:

1. **`parse_nibrs.py`** reads the FBI Annual Summary Report for 2019, sums Idaho drug arrest counts across all age groups → ~7,000 arrests
2. **`parse_teds.py`** reads the SAMHSA file, counts Idaho 2019 rows with drug admissions → ~500 admissions
3. **`fetch_acs.py`** fetches Idaho 2019 population from Census API → ~1.8M
4. **`02_data_processing.py`** merges and computes:
   - `arrest_rate = 7000 / 1,800,000 × 100,000 = ~389 per 100k`
   - `treatment_rate = 500 / 1,800,000 × 100,000 = ~28 per 100k`
   - `criminalization_index = 389 / 28 = ~14.1`
5. That row, combined with Idaho's demographic and political features, becomes one row in `panel_dataset.csv`
6. `04_modeling.py` reads all 377 rows and learns which features explain why Idaho's index is 50× higher than Connecticut's

### Why certain states are excluded

| Excluded | Reason | Where handled |
|---|---|---|
| Florida 2017–2021 | FBI stopped receiving Florida's NIBRS data — arrest numbers are artificially low | `02_data_processing.py` lines ~55–65 |
| Oregon (all years) | Oregon decriminalized drugs in 2020 and stopped reporting to federal treatment database — treatment rate is artificially low | `02_data_processing.py` lines ~70–75 |
| Idaho 2022, Illinois 2020–2021 | NIBRS transition artifacts — filtered in EDA and visualizations | `03_eda.py` lines ~25–30 |

---

## Current Results

### Model performance (5-fold cross-validated R²)

R² measures how much of the variation in state scores the model explains. 1.0 = perfect, 0.0 = no better than guessing the average.

| Model | R² | MAE | RMSE |
|---|---|---|---|
| OLS (linear regression) | 0.41 | — | — |
| Random Forest | 0.62 | — | — |
| Extra Trees | 0.65 | — | — |
| XGBoost | 0.65 | — | — |
| Gradient Boosting | **0.66** | 0.385 | 0.636 |
| Tuned XGBoost | **0.68** | — | — |

Full metrics are in `outputs/model_results.txt`.

### Top predictors (Gradient Boosting feature importance)

| Feature | Importance | Interpretation |
|---|---|---|
| `pres_x_incarceration` | 0.101 | Republican presidential vote × incarceration rate — the strongest combined signal |
| `region_northeast` | 0.083 | Northeast states score dramatically lower |
| `pres_vote_rep` | 0.072 | States that vote more Republican have higher indexes |
| `region_midwest` | 0.071 | Midwest also scores lower than South/Mountain West |
| `incarc_x_poverty` | 0.070 | Incarceration rate × poverty rate interaction |
| `incarceration_rate` | 0.067 | Higher incarceration → higher criminalization index |
| `overdose_death_rate` | 0.054 | Higher overdose deaths → slightly lower index |
| `marijuana_legal` | 0.044 | Legal recreational MJ → lower index |

### OLS significant coefficients (p < 0.05)

| Feature | Coefficient | Meaning |
|---|---|---|
| `incarceration_rate` | +1.43 | More incarcerated people → more criminalization |
| `pres_vote_rep` | +0.84 | More Republican presidential vote → more criminalization |
| `marijuana_legal` | −0.42 | Legal marijuana → less criminalization |
| `overdose_death_rate` | −0.92 | More overdose deaths → less criminalization |
| `poverty_rate` | −0.84 | More poverty → less criminalization |

---

## How to Modify or Improve

### Improving accuracy

**Add more features** — The model explains 68% of variation; 32% is unexplained. To add a new feature:
1. Add the column to `data/processed/panel_dataset.csv` by modifying `src/02_data_processing.py`
2. Add its column name to `EXTENDED_FEATURES` in `src/04_modeling.py` around lines 80–95
3. Re-run `02_data_processing.py` then `04_modeling.py`

**Add interaction terms** — The existing interactions (`pres_vote_rep × incarceration_rate`) are the top predictors. To add a new one, follow the same pattern in `src/04_modeling.py` around lines 140–160:
```python
df['new_interaction'] = df['col_a'] * df['col_b']
```

**Add state fixed effects** — The current models treat Idaho-2015 and Idaho-2016 as unrelated. Adding a dummy variable for each state would control for unmeasured state-level constants. In `02_data_processing.py`, add:
```python
state_dummies = pd.get_dummies(df['state'], prefix='fe', drop_first=True)
df = pd.concat([df, state_dummies], axis=1)
```
Then add the `fe_*` columns to `EXTENDED_FEATURES`.

**Increase tuning search** — In `src/04_modeling.py` around line 230, `n_iter=60`. Increasing to `150` searches more combinations at the cost of longer runtime (~15–20 min).

### Debugging poor performance

1. **Check R² dropped after a data change** — Re-run `03_eda.py` and verify `eda_summary.txt` still shows ~377 clean observations and a mean index near 1.29. If the count dropped or the mean shifted, something was excluded that shouldn't be.
2. **Check for outlier influence** — Run with `DROP_OUTLIERS = False` in the notebook. If R² changes dramatically, extreme values (Idaho 2019 at 14.1) are distorting the model.
3. **Check for correlated features** — The heatmap in `outputs/fig_distributions.png` shows feature correlations. Features with correlation >0.9 are redundant. Remove one of any near-identical pair from `EXTENDED_FEATURES`.
4. **Check the exclusions** — If a state you expect is missing, look in `02_data_processing.py` lines 55–75 (Florida/Oregon exclusion block) and `03_eda.py` lines 25–30 (additional filtering).

### Making changes safely

- **Never edit `panel_dataset.csv` by hand.** Fix the script and regenerate.
- **Use the notebook for experiments** before committing changes to the `.py` scripts.
- **The raw CSVs in `data/raw/` are the ground truth.** Don't delete them unless you are prepared to re-run the fetch/parse scripts.
- **Run `03_eda.py` after any data pipeline change** to sanity-check the observation count and index distribution before re-running the model.

---

This guide explains how to run all analysis for the Drug Policy Criminalization project using the consolidated Jupyter notebook.

## Setup

Make sure you have the required packages installed:
```bash
pip install pandas numpy scikit-learn statsmodels xgboost plotly seaborn matplotlib us
```

Ensure the processed dataset exists at: `./data/processed/panel_dataset.csv`

If not, run the data processing script first:
```bash
python src/02_data_processing.py
```

## Data Retrieval

Raw Data is not included in this repo. Current repo only works with processed data. If user wishes to personalize refer to Criminalization Vs. Treatment.md to find corresponding data.

## Quick Start

The notebook is configured with parameters at the top of the configuration cell. Look for these variables:

```python
DROP_OUTLIERS  = True          # Enable/disable outlier filtering
FEATURES_SET   = "extended"    # Feature set: "baseline" or "extended"
LOG_TRANSFORM  = True          # Log-transform the criminalization index
N_ESTIMATORS   = 400           # Number of trees in RF/GB models
RUN_TUNING     = True          # Enable/disable hyperparameter search
BEST_MODEL     = "auto"        # Final model selection
```

## Part 1: Baseline

Run the notebook with default settings:

1. Set parameters to:
   - `DROP_OUTLIERS = True`
   - `FEATURES_SET = "extended"`
   - `LOG_TRANSFORM = True`
   - `N_ESTIMATORS = 400`
   - `RUN_TUNING = False`
   - `BEST_MODEL = "auto"`

2. Run all cells in order

3. Record:
   - Model comparison R² table
   - Training time
   - Top predictor from feature importance chart

## Part 2: Data Quality Filtering

Test the effect of outlier removal:

1. Change only: `DROP_OUTLIERS = False`

2. Re-run from the filtering cell onward

3. Compare:
   - Criminalization index distribution
   - Choropleth map appearance
   - Model R² with and without outliers

**Why filtering matters:**
- Idaho 2022 and Florida 2017–2021 have known NIBRS reporting gaps
- Illinois 2020–2021 shows artifacts from the transition to NIBRS reporting
- Including these observations inflates variance and biases the index

## Part 3: Exploring EDA

Examine the exploratory visualizations:

1. Keep default settings
2. Run the EDA cell
3. Study the correlation heatmap — which demographic features correlate most with the index?
4. Note: states with high criminalization index vs. treatment orientation

**Questions to consider:**
- Which regions cluster together on the choropleth?
- Is the national arrest rate trending up or down?
- What is the median criminalization index, and how skewed is the distribution?

## Part 4: Feature Engineering

Test different feature sets:

**Baseline features** (7 features — demographics only):
1. Set `FEATURES_SET = "baseline"`
2. Run the feature engineering cell
3. Record: number of features, target mean/std

**Extended features** (28 features — demographics + policy + interactions):
1. Set `FEATURES_SET = "extended"`
2. Run the feature engineering cell
3. Compare feature counts and model R² in Part 5

**Log transformation:**
1. Set `LOG_TRANSFORM = False`
2. Re-run feature engineering and model training
3. Compare R² — log transform usually helps with the right-skewed index

**Expected observations:**
- Extended features add significant predictive power
- Log transform reduces RMSE and improves R² for tree models
- Political and structural features (incarceration rate, presidential vote) are often top predictors

## Part 5: Model Comparison

Compare five model types:

Suggested `N_ESTIMATORS` values to test:
- 100 — fast, rough baseline
- 400 — default, balanced
- 800 — slower, marginal improvement

For each:
1. Set `N_ESTIMATORS = 100` (or other value)
2. Run the model training cell
3. Record R², MAE, RMSE for all five models and training time

**Expected observations:**
- Tree models (RF, GB, XGB) outperform OLS significantly
- Extra Trees and Random Forest perform similarly
- XGBoost and Gradient Boosting are usually best
- More estimators help up to a point (~400–600 is usually enough)

## Part 6: Hyperparameter Tuning

Tune the best models from Part 5:

1. Set `RUN_TUNING = True`
2. Run the tuning cell (takes 5–15 minutes)
3. Compare tuned R² vs. default R²

**What gets tuned:**
- XGBoost: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, regularization
- Gradient Boosting: n_estimators, max_depth, learning_rate, subsample, max_features

**Expected observations:**
- Tuning typically improves R² by 0.02–0.08
- Best learning_rate is usually 0.03–0.07
- Deeper trees (max_depth=5–6) tend to work well for this dataset

## Part 7: Final Model

Combine your best settings from all experiments:

1. Set all parameters to your optimal configuration:
   - `DROP_OUTLIERS = True` (if it improved results in Part 2)
   - Best `FEATURES_SET` from Part 4
   - `LOG_TRANSFORM = True` (if it helped in Part 4)
   - Best `N_ESTIMATORS` from Part 5
   - `RUN_TUNING = True` (from Part 6)
   - `BEST_MODEL = "auto"` (or manually set to the winner)

2. Run all cells

3. Record final configuration and accuracy

Example optimal configuration:
```python
DROP_OUTLIERS  = True
FEATURES_SET   = "extended"
LOG_TRANSFORM  = True
N_ESTIMATORS   = 400
RUN_TUNING     = True
BEST_MODEL     = "auto"
```

## Tips for Best Results

1. **Always filter outliers** — Idaho/Florida/Illinois data artifacts inflate variance
2. **Use the extended feature set** — political and structural variables add real predictive power
3. **Log-transform the target** — criminalization index is right-skewed; log makes it more normal
4. **N_ESTIMATORS = 400** balances training time and accuracy
5. **Tuning helps** — especially for XGBoost; R² can jump by 0.05+

## Output Files

All outputs are saved to the `outputs/` directory:

| File | Description |
|------|-------------|
| `fig1_choropleth.html` | EDA choropleth (average index by state) |
| `fig2_time_trends.html` | EDA time trend (arrest vs treatment rate) |
| `fig_distributions.png` | Index distribution + correlation heatmap |
| `viz1_choropleth_final.html` | Final polished choropleth map |
| `viz2_time_trends_final.html` | Final dual-axis time trend |
| `viz3_feature_importance.html` | Interactive feature importance chart |
| `viz4_scatter_final.html` | Top predictor scatter with OLS trendline |
| `viz5_state_rankings.html` | State rankings bar chart |
| `fig3_feature_importance.png` | Static feature importance + predicted vs actual |
| `eda_summary.txt` | EDA text summary |
| `model_results.txt` | Model comparison metrics |

## Troubleshooting

**Out of memory during tuning:**
- Set `RUN_TUNING = False`
- Reduce `N_ESTIMATORS`

**Training too slow:**
- Set `N_ESTIMATORS = 100`
- Set `RUN_TUNING = False`
- Set `FEATURES_SET = "baseline"`

**Poor R² (below 0.4):**
- Enable `DROP_OUTLIERS = True`
- Switch to `FEATURES_SET = "extended"`
- Enable `LOG_TRANSFORM = True`
- Run tuning with `RUN_TUNING = True`

## Quick Experiment Loop

```python
# Example: Test multiple feature sets
for fset in ["baseline", "extended"]:
    print(f"\n{'='*60}")
    print(f"Testing FEATURES_SET = '{fset}'")
    print(f"{'='*60}")
    # Set FEATURES_SET = fset and re-run feature engineering + modeling cells
```

Good luck with your experiments!
