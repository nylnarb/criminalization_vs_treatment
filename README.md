# Drug Policy Criminalization — Experiment Guide

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
