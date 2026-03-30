"""
Modeling: Predicting the Criminalization Index
Three models compared:
  1. OLS Linear Regression (baseline)
  2. Random Forest
  3. Gradient Boosting (XGBoost)

Target: log(criminalization_index)  -- log-transform to handle right skew
Features: poverty_rate, median_income, unemployment_rate,
          pct_white, pct_black, pct_hispanic, year

Outputs:
  outputs/model_results.txt         -- performance metrics + OLS coefficients
  outputs/fig3_feature_importance.png  -- RF + XGB feature importance
  outputs/fig4_scatter_regression.png  -- predicted vs actual + top predictor scatter
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(BASE, "data/processed/panel_dataset.csv")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA)

# ── Drop known data-quality outliers (same as EDA) ───────────────────────────
bad_mask = (
    ((df['state'] == 'Idaho')    & (df['year'] == 2022)) |
    ((df['state'] == 'Florida')  & (df['year'].isin([2017, 2018, 2019, 2020, 2021]))) |
    ((df['state'] == 'Illinois') & (df['year'].isin([2020, 2021])))
)
df = df[~bad_mask].copy()

FEATURES = [
    # Demographic / socioeconomic
    'poverty_rate', 'median_income', 'unemployment_rate',
    'pct_white', 'pct_black', 'pct_hispanic',
    # Structural
    'overdose_death_rate', 'incarceration_rate',
    'marijuana_legal', 'republican_gov',
    # Political lean & continuity
    'pres_vote_rep', 'gov_streak',
    # Legislative composition
    'senate_rep_pct', 'house_rep_pct',
    # Government alignment dummies (ref = unified_rep)
    'align_unified_dem', 'align_div_rep_gov', 'align_div_dem_gov',
    'year',
]
TARGET = 'criminalization_index'

model_df = df.dropna(subset=FEATURES + [TARGET]).copy()
model_df['log_index'] = np.log(model_df[TARGET])

# ── Region dummies (4 US Census regions → 3 dummies, ref = South) ────────────
NORTHEAST = {'Connecticut','Maine','Maryland','Massachusetts','New Hampshire',
             'New Jersey','New York','Pennsylvania','Rhode Island','Vermont','Delaware'}
MIDWEST   = {'Illinois','Indiana','Iowa','Kansas','Michigan','Minnesota',
             'Missouri','Nebraska','North Dakota','Ohio','South Dakota','Wisconsin'}
WEST      = {'Alaska','Arizona','California','Colorado','Hawaii','Idaho',
             'Montana','Nevada','New Mexico','Utah','Washington','Wyoming'}
model_df['region_northeast'] = model_df['state'].isin(NORTHEAST).astype(float)
model_df['region_midwest']   = model_df['state'].isin(MIDWEST).astype(float)
model_df['region_west']      = model_df['state'].isin(WEST).astype(float)

# ── Interaction terms ─────────────────────────────────────────────────────────
model_df['pres_x_overdose']         = model_df['pres_vote_rep'] * model_df['overdose_death_rate']
model_df['pres_x_incarceration']    = model_df['pres_vote_rep'] * model_df['incarceration_rate']
model_df['rep_pct_x_incarceration'] = model_df['senate_rep_pct'] * model_df['incarceration_rate']
model_df['incarc_x_poverty']        = model_df['incarceration_rate'] * model_df['poverty_rate']
model_df['overdose_x_mj']           = model_df['overdose_death_rate'] * model_df['marijuana_legal']
model_df['incarc_x_midwest']        = model_df['incarceration_rate'] * model_df['region_midwest']
model_df['incarc_x_south']          = (
    model_df['incarceration_rate'] *
    (1 - model_df['region_northeast'] - model_df['region_midwest'] - model_df['region_west'])
)

FEATURES_ALL = FEATURES + [
    'region_northeast', 'region_midwest', 'region_west',
    'pres_x_overdose', 'pres_x_incarceration', 'rep_pct_x_incarceration',
    'incarc_x_poverty', 'overdose_x_mj', 'incarc_x_midwest', 'incarc_x_south',
]

X = model_df[FEATURES_ALL].values
y = model_df['log_index'].values

print(f"Modeling on {len(model_df)} complete observations, {len(FEATURES_ALL)} features")
print(f"Target: log(criminalization_index)  |  mean={y.mean():.3f}  std={y.std():.3f}\n")

# ── Cross-validation setup ────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_metrics(model, X, y, label):
    r2s   = cross_val_score(model, X, y, cv=kf, scoring='r2')
    maes  = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    rmses = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))
    print(f"{label}:")
    print(f"  R²   = {r2s.mean():.3f}  (+/- {r2s.std():.3f})")
    print(f"  MAE  = {maes.mean():.3f}  (+/- {maes.std():.3f})")
    print(f"  RMSE = {rmses.mean():.3f}  (+/- {rmses.std():.3f})\n")
    return {'label': label, 'R2': r2s.mean(), 'R2_std': r2s.std(),
            'MAE': maes.mean(), 'RMSE': rmses.mean()}

# ── 1. OLS (statsmodels for coefficients + p-values) ─────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_sm = sm.add_constant(X_scaled)
ols_model = sm.OLS(y, X_sm).fit()

print("=" * 55)
print("MODEL COMPARISON (5-Fold CV on log-transformed index)")
print("=" * 55 + "\n")

ols_sk = LinearRegression()
rf     = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=3,
                                random_state=42, n_jobs=-1)
et     = ExtraTreesRegressor(n_estimators=400, max_depth=6, min_samples_leaf=3,
                              random_state=42, n_jobs=-1)
gb     = GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.04,
                                    subsample=0.7, min_samples_leaf=4, random_state=42)
xgb    = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.04,
                       subsample=0.7, colsample_bytree=0.8, min_child_weight=4,
                       random_state=42, n_jobs=-1, verbosity=0)

results = []
results.append(cv_metrics(ols_sk, X_scaled, y, "OLS Linear Regression"))
results.append(cv_metrics(rf,     X,        y, "Random Forest"))
results.append(cv_metrics(et,     X,        y, "Extra Trees"))
results.append(cv_metrics(gb,     X,        y, "Gradient Boosting"))
results.append(cv_metrics(xgb,    X,        y, "XGBoost"))

# ── OLS coefficients ──────────────────────────────────────────────────────────
print("\n--- OLS Coefficients (standardized features) ---")
feat_names = ['intercept'] + FEATURES_ALL
coef_df = pd.DataFrame({
    'feature':   feat_names,
    'coef':      ols_model.params,
    'std_err':   ols_model.bse,
    't_stat':    ols_model.tvalues,
    'p_value':   ols_model.pvalues,
})
print(coef_df.to_string(index=False))
print(f"\nOLS R² (in-sample): {ols_model.rsquared:.3f}")

# ── Hyperparameter search: XGBoost ───────────────────────────────────────────
print("Running RandomizedSearchCV for XGBoost...")
xgb_param_dist = {
    'n_estimators':      [300, 400, 500, 700],
    'max_depth':         [3, 4, 5, 6],
    'learning_rate':     [0.02, 0.03, 0.05, 0.07, 0.1],
    'subsample':         [0.6, 0.7, 0.8],
    'colsample_bytree':  [0.6, 0.7, 0.8, 1.0],
    'min_child_weight':  [3, 4, 5, 6],
    'reg_alpha':         [0, 0.01, 0.1, 0.5],
    'reg_lambda':        [0.5, 1, 2, 5],
}
xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    param_distributions=xgb_param_dist,
    n_iter=60, cv=kf, scoring='r2', random_state=42, n_jobs=1,
)
xgb_search.fit(X, y)
print(f"Best XGB params: {xgb_search.best_params_}")
xgb_tuned_r2 = xgb_search.best_score_
print(f"Tuned XGBoost R² (CV): {xgb_tuned_r2:.3f}\n")

# Also tune GB
print("Running RandomizedSearchCV for Gradient Boosting...")
gb_param_dist = {
    'n_estimators':     [300, 500, 700, 1000],
    'max_depth':        [3, 4, 5],
    'learning_rate':    [0.02, 0.03, 0.05, 0.07],
    'subsample':        [0.6, 0.7, 0.8],
    'min_samples_leaf': [3, 4, 5, 6, 8],
    'max_features':     [0.6, 0.7, 0.8, 'sqrt'],
}
gb_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_distributions=gb_param_dist,
    n_iter=60, cv=kf, scoring='r2', random_state=42, n_jobs=-1,
)
gb_search.fit(X, y)
print(f"Best GB params: {gb_search.best_params_}")
gb_tuned_r2 = gb_search.best_score_
print(f"Tuned GB R² (CV): {gb_tuned_r2:.3f}\n")

best_r2 = max(xgb_tuned_r2, gb_tuned_r2)
best_label = 'XGBoost (tuned)' if xgb_tuned_r2 >= gb_tuned_r2 else 'Gradient Boosting (tuned)'
results.append({'label': best_label, 'R2': best_r2, 'R2_std': 0.0, 'MAE': 0.0, 'RMSE': 0.0})
best_model = xgb_search.best_estimator_ if xgb_tuned_r2 >= gb_tuned_r2 else gb_search.best_estimator_
print(f"\nBest model: {best_label}  R²={best_r2:.3f}")

# ── Fit final models on full data for feature importance ─────────────────────
rf.fit(X, y)
best_model.fit(X, y)
gb = best_model  # use tuned XGB for plots

# ── Fig 3: Feature importance (RF + GB side by side) ─────────────────────────
feat_labels = [f.replace('_', ' ') for f in FEATURES_ALL]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, model, title, color in [
    (axes[0], rf, 'Random Forest',       'steelblue'),
    (axes[1], gb, 'Gradient Boosting',   'darkorange'),
]:
    imp = model.feature_importances_
    order = np.argsort(imp)
    ax.barh([feat_labels[i] for i in order], imp[order], color=color, alpha=0.85)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{title}\nFeature Importance')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Predictors of Criminalization Index', fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_feature_importance.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: fig3_feature_importance.png")

# ── Fig 4: Predicted vs Actual + top predictor scatter ───────────────────────
gb_preds = gb.predict(X)

top_feat_idx = np.argmax(gb.feature_importances_)
top_feat     = FEATURES_ALL[top_feat_idx]
top_label    = feat_labels[top_feat_idx]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: predicted vs actual
axes[0].scatter(y, gb_preds, alpha=0.5, color='steelblue', s=25, edgecolors='none')
lims = [min(y.min(), gb_preds.min()) - 0.2, max(y.max(), gb_preds.max()) + 0.2]
axes[0].plot(lims, lims, 'r--', linewidth=1.2, label='Perfect fit')
axes[0].set_xlabel('Actual log(Criminalization Index)')
axes[0].set_ylabel('Predicted')
axes[0].set_title('Gradient Boosting: Predicted vs Actual')
r2_full = r2_score(y, gb_preds)
axes[0].text(0.05, 0.92, f'R² = {r2_full:.3f}', transform=axes[0].transAxes,
             fontsize=10, color='black')
axes[0].legend()

# Right: top predictor vs log(index) with regression line
x_top = model_df[top_feat].values
m, b  = np.polyfit(x_top, y, 1)
x_line = np.linspace(x_top.min(), x_top.max(), 100)
axes[1].scatter(x_top, y, alpha=0.45, color='darkorange', s=25, edgecolors='none')
axes[1].plot(x_line, m * x_line + b, 'r-', linewidth=2)
axes[1].set_xlabel(top_label.title())
axes[1].set_ylabel('log(Criminalization Index)')
axes[1].set_title(f'Top Predictor: {top_label.title()}')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_scatter_regression.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig4_scatter_regression.png")

# ── Save full results to text ─────────────────────────────────────────────────
res_df = pd.DataFrame(results)[['label','R2','R2_std','MAE','RMSE']]
lines = [
    "=== MODEL RESULTS ===\n",
    res_df.to_string(index=False),
    "\n\n--- OLS Coefficients (standardized) ---",
    coef_df.to_string(index=False),
    f"\nOLS in-sample R²: {ols_model.rsquared:.3f}",
    f"\nGradient Boosting top feature: {top_feat}",
    "\n--- RF Feature Importance ---",
    pd.DataFrame({'feature': FEATURES_ALL,
                  'importance': rf.feature_importances_})
    .sort_values('importance', ascending=False).to_string(index=False),
    "\n--- GB Feature Importance ---",
    pd.DataFrame({'feature': FEATURES_ALL,
                  'importance': gb.feature_importances_})
    .sort_values('importance', ascending=False).to_string(index=False),
]
with open(os.path.join(OUT_DIR, "model_results.txt"), "w") as f:
    f.write("\n".join(lines))
print("Saved: model_results.txt")
