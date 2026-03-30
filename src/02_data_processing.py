"""
Merges NIBRS drug arrests + SAMHSA TEDS treatment admissions + Census ACS
+ policy features (marijuana, governor party, overdose deaths, incarceration rate)
into a clean state-year panel dataset (2015-2022).

Outputs: data/processed/panel_dataset.csv

Key computed variables:
  arrest_rate           = drug_arrests / population * 100000
  treatment_rate        = drug_treatment_admissions / population * 100000
  overdose_death_rate   = overdose_deaths / population * 100000
  criminalization_index = arrest_rate / treatment_rate
"""

import os
import numpy as np
import pandas as pd

BASE    = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

def clean_state(s):
    if not isinstance(s, str):
        return np.nan
    return s.strip().title()

# ── Load raw files ─────────────────────────────────────────────────────────────
print("Loading raw datasets...")

nibrs  = pd.read_csv(os.path.join(BASE, "data/raw/nibrs/nibrs_2015_2022_combined.csv"))
teds   = pd.read_csv(os.path.join(BASE, "data/raw/teds/teds_2015_2022_combined.csv"))
acs    = pd.read_csv(os.path.join(BASE, "data/raw/acs/acs_2015_2022_combined.csv"))
mj     = pd.read_csv(os.path.join(BASE, "data/raw/policy/recreational_marijuana.csv"))
gov    = pd.read_csv(os.path.join(BASE, "data/raw/policy/governor_party.csv"))
cdc    = pd.read_csv(os.path.join(BASE, "data/raw/cdc/cdc_overdose_2015_2022.csv"))
bjs    = pd.read_csv(os.path.join(BASE, "data/raw/bjs/bjs_incarceration_2015_2022.csv"))
polft  = pd.read_csv(os.path.join(BASE, "data/raw/policy/political_features.csv"))
legft  = pd.read_csv(os.path.join(BASE, "data/raw/policy/legislature_control.csv"))

for name, frame in [('NIBRS',nibrs),('TEDS',teds),('ACS',acs),('MJ',mj),
                     ('GOV',gov),('CDC',cdc),('BJS',bjs),('POLFT',polft),('LEGFT',legft)]:
    print(f"  {name}: {len(frame)} rows")

# ── Standardize state names ────────────────────────────────────────────────────
for frame in [nibrs, teds, acs, mj, gov, bjs, polft, legft]:
    frame['state'] = frame['state'].apply(clean_state)

# CDC uses full state names in 'state' column
cdc['state'] = cdc['state'].apply(clean_state)

FIFTY_STATES = {
    'Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut',
    'Delaware','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa',
    'Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan',
    'Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada',
    'New Hampshire','New Jersey','New Mexico','New York','North Carolina',
    'North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island',
    'South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont',
    'Virginia','Washington','West Virginia','Wisconsin','Wyoming',
}

acs = acs[acs['state'].isin(FIFTY_STATES)]
cdc = cdc[cdc['state'].isin(FIFTY_STATES)]

# ── Flag known bad NIBRS data: Florida ASR 2017-2019 ──────────────────────────
fl_asr_mask = (nibrs['state'] == 'Florida') & (nibrs['source'] == 'ASR')
nibrs.loc[fl_asr_mask, 'drug_arrests'] = np.nan

# ── Merge core datasets ────────────────────────────────────────────────────────
print("\nMerging datasets...")
panel = pd.merge(
    nibrs[['state','year','drug_arrests']],
    teds[['state','year','drug_treatment_admissions']],
    on=['state','year'], how='outer',
)
panel = pd.merge(
    panel,
    acs[['state','year','population','poverty_rate','median_income',
         'unemployment_rate','pct_white','pct_black','pct_hispanic']],
    on=['state','year'], how='left',
)

# ── Merge new policy/structural features ──────────────────────────────────────
panel = pd.merge(panel, mj[['state','year','marijuana_legal']],
                 on=['state','year'], how='left')
panel = pd.merge(panel, gov[['state','year','republican_gov']],
                 on=['state','year'], how='left')
panel = pd.merge(panel, cdc[['state','year','overdose_deaths']],
                 on=['state','year'], how='left')
panel = pd.merge(panel, bjs[['state','year','incarceration_rate']],
                 on=['state','year'], how='left')
panel = pd.merge(panel, polft[['state','year','gov_streak','pres_vote_rep']],
                 on=['state','year'], how='left')
panel = pd.merge(panel, legft[['state','year','senate_rep_pct','house_rep_pct',
                                'gov_leg_alignment','align_unified_dem',
                                'align_div_rep_gov','align_div_dem_gov']],
                 on=['state','year'], how='left')

# ── Exclude Oregon ─────────────────────────────────────────────────────────────
panel = panel[panel['state'] != 'Oregon'].copy()

# ── Compute per-100k rates ─────────────────────────────────────────────────────
panel['arrest_rate']        = panel['drug_arrests'] / panel['population'] * 100_000
panel['treatment_rate']     = panel['drug_treatment_admissions'] / panel['population'] * 100_000
panel['overdose_death_rate'] = panel['overdose_deaths'] / panel['population'] * 100_000

# ── Criminalization Index ──────────────────────────────────────────────────────
panel['criminalization_index'] = np.where(
    (panel['treatment_rate'] > 0) & panel['arrest_rate'].notna(),
    panel['arrest_rate'] / panel['treatment_rate'],
    np.nan,
)

# ── Sort and save ──────────────────────────────────────────────────────────────
panel = panel.sort_values(['state','year']).reset_index(drop=True)
out_path = os.path.join(OUT_DIR, "panel_dataset.csv")
panel.to_csv(out_path, index=False)

# ── Summary ────────────────────────────────────────────────────────────────────
ALL_FEATURES = ['arrest_rate','treatment_rate','criminalization_index',
                'population','poverty_rate','overdose_death_rate',
                'incarceration_rate','marijuana_legal','republican_gov',
                'gov_streak','pres_vote_rep','senate_rep_pct','house_rep_pct']
complete = panel.dropna(subset=ALL_FEATURES)

print(f"\nSaved: {out_path}")
print(f"Total rows:    {len(panel)}")
print(f"Complete rows: {len(complete)}  (all features present)")
print(f"States:        {panel['state'].nunique()}")
print(f"Years:         {sorted(panel['year'].unique())}")

print("\nCriminalization Index summary (complete rows):")
print(complete['criminalization_index'].describe().round(3).to_string())

print("\nMissing value counts (full panel):")
print(panel[ALL_FEATURES + ['median_income','unemployment_rate',
            'pct_white','pct_black','pct_hispanic']].isna().sum().to_string())
