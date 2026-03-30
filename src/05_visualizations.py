"""
Final presentation-ready visualizations.

Outputs:
  outputs/viz1_choropleth_final.html     -- polished choropleth map
  outputs/viz2_time_trends_final.html    -- arrest vs treatment trend lines
  outputs/viz3_feature_importance.html   -- interactive feature importance
  outputs/viz4_scatter_final.html        -- top predictor scatter with regression
  outputs/viz5_state_rankings.html       -- horizontal bar chart of state averages
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import us

BASE    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(BASE, "data/processed/panel_dataset.csv")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA)

# ── Drop known data-quality outliers ─────────────────────────────────────────
bad_mask = (
    (df['state'] == 'Idaho') |
    (df['state'] == 'Florida') |
    ((df['state'] == 'Illinois') & (df['year'].isin([2020,2021])))
)
clean = df[~bad_mask].copy()
complete = clean.dropna(subset=['criminalization_index','arrest_rate',
                                 'treatment_rate','poverty_rate','population'])

FEATURES = ['poverty_rate','median_income','unemployment_rate',
            'pct_white','pct_black','pct_hispanic','year']
feat_labels = {
    'poverty_rate':       'Poverty Rate',
    'median_income':      'Median Income',
    'unemployment_rate':  'Unemployment Rate',
    'pct_white':          '% White',
    'pct_black':          '% Black',
    'pct_hispanic':       '% Hispanic',
    'year':               'Year',
}

# ── Fit final GB model ────────────────────────────────────────────────────────
model_df = complete.dropna(subset=FEATURES).copy()
model_df['log_index'] = np.log(model_df['criminalization_index'])
X = model_df[FEATURES].values
y = model_df['log_index'].values
gb = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                learning_rate=0.05, subsample=0.8, random_state=42)
gb.fit(X, y)

# ── State averages ────────────────────────────────────────────────────────────
state_avg = (complete.groupby('state')
             .agg(avg_index=('criminalization_index','mean'),
                  avg_arrest=('arrest_rate','mean'),
                  avg_treatment=('treatment_rate','mean'))
             .reset_index())
state_avg['code'] = state_avg['state'].apply(
    lambda s: us.states.lookup(s).abbr if us.states.lookup(s) else None)
state_avg = state_avg.dropna(subset=['code'])
state_avg['orientation'] = state_avg['avg_index'].apply(
    lambda x: 'Criminalization-oriented' if x >= 1 else 'Treatment-oriented')

# ── VIZ 1: Choropleth ─────────────────────────────────────────────────────────
fig1 = px.choropleth(
    state_avg,
    locations='code',
    locationmode='USA-states',
    color='avg_index',
    scope='usa',
    color_continuous_scale='RdYlGn_r',
    hover_name='state',
    hover_data={
        'code': False,
        'avg_index': ':.2f',
        'avg_arrest': ':.1f',
        'avg_treatment': ':.1f',
    },
    labels={
        'avg_index':     'Criminalization Index',
        'avg_arrest':    'Avg Arrest Rate/100k',
        'avg_treatment': 'Avg Treatment Rate/100k',
    },
    title='<b>Drug Policy Orientation by State</b><br>'
          '<sup>Average Criminalization Index 2015–2022 | '
          'Higher = more arrests relative to treatment admissions</sup>',
)
fig1.update_layout(
    coloraxis_colorbar=dict(title='Index', tickformat='.1f'),
    margin=dict(l=0, r=0, t=70, b=0),
    font=dict(family='Arial', size=12),
)
fig1.write_html(os.path.join(OUT_DIR, "viz1_choropleth_final.html"))
print("Saved: viz1_choropleth_final.html")

# ── VIZ 2: Time trends ────────────────────────────────────────────────────────
national = (complete.groupby('year')
            .apply(lambda g: pd.Series({
                'arrest_rate':    np.average(g['arrest_rate'],    weights=g['population']),
                'treatment_rate': np.average(g['treatment_rate'], weights=g['population']),
                'avg_index':      np.average(g['criminalization_index'], weights=g['population']),
            }), include_groups=False)
            .reset_index())

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Scatter(
    x=national['year'], y=national['arrest_rate'],
    name='Drug Arrest Rate', mode='lines+markers',
    line=dict(color='#d62728', width=2.5), marker=dict(size=8),
), secondary_y=False)
fig2.add_trace(go.Scatter(
    x=national['year'], y=national['treatment_rate'],
    name='Treatment Admission Rate', mode='lines+markers',
    line=dict(color='#1f77b4', width=2.5), marker=dict(size=8),
), secondary_y=False)
fig2.add_trace(go.Scatter(
    x=national['year'], y=national['avg_index'],
    name='Criminalization Index', mode='lines+markers',
    line=dict(color='#2ca02c', width=2, dash='dot'), marker=dict(size=7),
), secondary_y=True)
fig2.add_hline(y=1.0, secondary_y=True,
               line=dict(color='gray', dash='dash', width=1),
               annotation_text='Index = 1.0', annotation_position='right')
fig2.update_layout(
    title='<b>National Drug Arrest vs. Treatment Rates Over Time</b><br>'
          '<sup>Population-weighted average, 49 states, 2015–2022</sup>',
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99),
    font=dict(family='Arial', size=12),
)
fig2.update_yaxes(title_text='Rate per 100,000', secondary_y=False)
fig2.update_yaxes(title_text='Criminalization Index', secondary_y=True)
fig2.write_html(os.path.join(OUT_DIR, "viz2_time_trends_final.html"))
print("Saved: viz2_time_trends_final.html")

# ── VIZ 3: Feature importance (interactive) ───────────────────────────────────
imp_df = pd.DataFrame({
    'Feature':    [feat_labels[f] for f in FEATURES],
    'Importance': gb.feature_importances_,
}).sort_values('Importance')

fig3 = px.bar(
    imp_df, x='Importance', y='Feature', orientation='h',
    color='Importance', color_continuous_scale='Blues',
    title='<b>What Predicts Criminalization Index?</b><br>'
          '<sup>Gradient Boosting feature importance (higher = stronger predictor)</sup>',
    labels={'Importance': 'Feature Importance', 'Feature': ''},
)
fig3.update_layout(
    coloraxis_showscale=False,
    font=dict(family='Arial', size=12),
    margin=dict(l=150, r=40, t=80, b=40),
)
fig3.write_html(os.path.join(OUT_DIR, "viz3_feature_importance.html"))
print("Saved: viz3_feature_importance.html")

# ── VIZ 4: Scatter — top predictor vs index ───────────────────────────────────
top_feat = FEATURES[np.argmax(gb.feature_importances_)]
top_label = feat_labels[top_feat]

scatter_df = complete.dropna(subset=[top_feat, 'criminalization_index']).copy()
scatter_df['log_index'] = np.log(scatter_df['criminalization_index'])

fig4 = px.scatter(
    scatter_df, x=top_feat, y='criminalization_index',
    color='state', hover_name='state',
    hover_data={'year': True, 'arrest_rate': ':.1f',
                'treatment_rate': ':.1f', 'state': False},
    log_y=True,
    trendline='ols',
    labels={
        top_feat: top_label,
        'criminalization_index': 'Criminalization Index (log scale)',
    },
    title=f'<b>Criminalization Index vs. {top_label}</b><br>'
          '<sup>Each point = one state-year | log scale | OLS trend line</sup>',
)
fig4.update_traces(marker=dict(size=6, opacity=0.6))
fig4.update_layout(
    showlegend=False,
    font=dict(family='Arial', size=12),
)
fig4.write_html(os.path.join(OUT_DIR, "viz4_scatter_final.html"))
print("Saved: viz4_scatter_final.html")

# ── VIZ 5: State rankings bar chart ──────────────────────────────────────────
ranked = state_avg.sort_values('avg_index', ascending=True)
ranked['color'] = ranked['avg_index'].apply(
    lambda x: '#d62728' if x >= 1 else '#1f77b4')

fig5 = go.Figure(go.Bar(
    x=ranked['avg_index'],
    y=ranked['state'],
    orientation='h',
    marker_color=ranked['color'],
    text=ranked['avg_index'].round(2),
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>Index: %{x:.2f}<extra></extra>',
))
fig5.add_vline(x=1.0, line=dict(color='black', dash='dash', width=1.5),
               annotation_text='Index = 1.0', annotation_position='top right')
fig5.update_layout(
    title='<b>Average Criminalization Index by State (2015–2022)</b><br>'
          '<sup>Red = more arrests than treatment | Blue = more treatment than arrests</sup>',
    xaxis_title='Criminalization Index',
    yaxis_title='',
    height=1100,
    margin=dict(l=150, r=80, t=80, b=40),
    font=dict(family='Arial', size=11),
)
fig5.write_html(os.path.join(OUT_DIR, "viz5_state_rankings.html"))
print("Saved: viz5_state_rankings.html")
print("\nAll visualizations complete.")
