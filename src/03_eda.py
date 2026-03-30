"""
Exploratory Data Analysis
Produces:
  outputs/fig1_choropleth.html         -- U.S. choropleth of avg Criminalization Index
  outputs/fig2_time_trends.html        -- national arrest vs treatment rate over time
  outputs/fig_distributions.png        -- index distribution + correlation heatmap
  outputs/eda_summary.txt              -- printed stats summary
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

BASE    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(BASE, "data/processed/panel_dataset.csv")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA)

# ── Drop known data-quality outliers ──────────────────────────────────────────
bad_mask = (
    ((df['state'] == 'Idaho')    & (df['year'] == 2022)) |
    ((df['state'] == 'Florida')  & (df['year'].isin([2017,2018,2019,2020,2021]))) |
    ((df['state'] == 'Illinois') & (df['year'].isin([2020,2021])))
)
clean = df[~bad_mask].copy()
complete = clean.dropna(subset=['criminalization_index','arrest_rate',
                                 'treatment_rate','poverty_rate','population'])
print(f"Clean complete rows for analysis: {len(complete)}")

# ── Fig 1: Choropleth — average Criminalization Index by state ─────────────────
state_avg = (complete.groupby('state')['criminalization_index']
             .mean().reset_index()
             .rename(columns={'criminalization_index': 'avg_index'}))

# Plotly needs 2-letter state codes
import us
state_avg['code'] = state_avg['state'].apply(
    lambda s: us.states.lookup(s).abbr if us.states.lookup(s) else None)
state_avg = state_avg.dropna(subset=['code'])

fig1 = px.choropleth(
    state_avg,
    locations='code',
    locationmode='USA-states',
    color='avg_index',
    scope='usa',
    color_continuous_scale='RdYlGn_r',
    labels={'avg_index': 'Avg Criminalization Index'},
    title='Average Criminalization Index by State (2015–2022)<br>'
          '<sup>Higher = more arrests relative to treatment admissions</sup>',
)
fig1.update_layout(
    coloraxis_colorbar=dict(title='Index'),
    margin=dict(l=0, r=0, t=60, b=0),
)
fig1.write_html(os.path.join(OUT_DIR, "fig1_choropleth.html"))
print("Saved: fig1_choropleth.html")

# ── Fig 2: Time trend — national arrest rate vs treatment rate ─────────────────
national = (complete.groupby('year')
            .apply(lambda g: pd.Series({
                'arrest_rate':    np.average(g['arrest_rate'],    weights=g['population']),
                'treatment_rate': np.average(g['treatment_rate'], weights=g['population']),
            }))
            .reset_index())

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=national['year'], y=national['arrest_rate'],
    mode='lines+markers', name='Drug Arrest Rate',
    line=dict(color='#d62728', width=2.5),
    marker=dict(size=7),
))
fig2.add_trace(go.Scatter(
    x=national['year'], y=national['treatment_rate'],
    mode='lines+markers', name='Treatment Admission Rate',
    line=dict(color='#1f77b4', width=2.5),
    marker=dict(size=7),
))
fig2.update_layout(
    title='National Drug Arrest Rate vs. Treatment Admission Rate (per 100k)<br>'
          '<sup>Population-weighted average across 49 states, 2015–2022</sup>',
    xaxis_title='Year',
    yaxis_title='Rate per 100,000',
    legend=dict(x=0.01, y=0.99),
    hovermode='x unified',
)
fig2.write_html(os.path.join(OUT_DIR, "fig2_time_trends.html"))
print("Saved: fig2_time_trends.html")

# ── Fig 3: Distribution + correlation heatmap (static PNG) ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: distribution of Criminalization Index
axes[0].hist(complete['criminalization_index'], bins=40,
             color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(complete['criminalization_index'].median(), color='red',
                linestyle='--', label=f"Median = {complete['criminalization_index'].median():.2f}")
axes[0].axvline(1.0, color='black', linestyle=':', linewidth=1.2, label='Index = 1.0')
axes[0].set_xlabel('Criminalization Index')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Criminalization Index')
axes[0].legend()

# Right: correlation heatmap
corr_cols = ['criminalization_index','arrest_rate','treatment_rate',
             'poverty_rate','median_income','unemployment_rate',
             'pct_white','pct_black','pct_hispanic']
corr = complete[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=axes[1],
            annot_kws={'size': 8},
            xticklabels=[c.replace('_',' ') for c in corr_cols],
            yticklabels=[c.replace('_',' ') for c in corr_cols])
axes[1].set_title('Correlation Matrix')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_distributions.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_distributions.png")

# ── Text summary ───────────────────────────────────────────────────────────────
summary_lines = []
summary_lines.append("=== EDA SUMMARY ===\n")
summary_lines.append(f"Clean observations: {len(complete)}")
summary_lines.append(f"States: {complete['state'].nunique()}  |  Years: 2015-2022\n")

summary_lines.append("--- Criminalization Index ---")
summary_lines.append(complete['criminalization_index'].describe().round(3).to_string())

summary_lines.append("\n--- Top 10 most criminalizing state-years ---")
top10 = complete.nlargest(10,'criminalization_index')[
    ['state','year','arrest_rate','treatment_rate','criminalization_index']]
summary_lines.append(top10.to_string(index=False))

summary_lines.append("\n--- Top 10 most treatment-oriented state-years ---")
bot10 = complete.nsmallest(10,'criminalization_index')[
    ['state','year','arrest_rate','treatment_rate','criminalization_index']]
summary_lines.append(bot10.to_string(index=False))

summary_lines.append("\n--- Average Criminalization Index by state (2015-2022) ---")
summary_lines.append(state_avg.sort_values('avg_index', ascending=False)
                     [['state','avg_index']].to_string(index=False))

summary_lines.append("\n--- National trend (population-weighted) ---")
summary_lines.append(national.to_string(index=False))

txt = "\n".join(summary_lines)
print("\n" + txt)
with open(os.path.join(OUT_DIR, "eda_summary.txt"), "w") as f:
    f.write(txt)
print("\nSaved: eda_summary.txt")
