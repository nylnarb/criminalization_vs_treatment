"""
Builds state legislative control features 2015-2022.

Sources:
  2015-2021: psthomas/state-partisan-composition (GitHub CSV, already downloaded)
  2022:      Hardcoded from NCSL State Partisan Composition Table, June 2022

For each state-year extracts/computes:
  senate_dem, senate_rep  -- seat counts
  house_dem, house_rep    -- seat counts
  leg_control             -- Rep / Dem / Split / Divided
  senate_rep_pct, house_rep_pct
  unified_rep, unified_dem
  gov_leg_alignment: unified_rep | unified_dem | div_rep_gov | div_dem_gov | mixed

Saves: data/raw/policy/legislature_control.csv
"""

import os
import pandas as pd
import numpy as np

BASE    = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE, "data/raw/policy")

STATES_49 = {
    'Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut',
    'Delaware','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa',
    'Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan',
    'Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada',
    'New Hampshire','New Jersey','New Mexico','New York','North Carolina',
    'North Dakota','Ohio','Oklahoma','Pennsylvania','Rhode Island',
    'South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont',
    'Virginia','Washington','West Virginia','Wisconsin','Wyoming',
}

# ── 2022 data hardcoded from NCSL State Partisan Composition Table, June 1 2022 ──
# Fields: senate_dem, senate_rep, house_dem, house_rep, leg_control
DATA_2022 = {
    'Alabama':        (8,  27,  28,  73,  'Rep'),
    'Alaska':         (7,  13,  15,  21,  'Rep'),
    'Arizona':        (14, 16,  29,  31,  'Rep'),
    'Arkansas':       (7,  27,  22,  78,  'Rep'),
    'California':     (31, 9,   58,  19,  'Dem'),
    'Colorado':       (20, 15,  41,  24,  'Dem'),
    'Connecticut':    (23, 13,  97,  54,  'Dem'),
    'Delaware':       (14, 7,   26,  15,  'Dem'),
    'Florida':        (16, 24,  42,  76,  'Rep'),
    'Georgia':        (22, 34,  77,  103, 'Rep'),
    'Hawaii':         (24, 1,   47,  4,   'Dem'),
    'Idaho':          (7,  28,  12,  58,  'Rep'),
    'Illinois':       (41, 18,  73,  45,  'Dem'),
    'Indiana':        (11, 39,  29,  71,  'Rep'),
    'Iowa':           (18, 32,  40,  60,  'Rep'),
    'Kansas':         (11, 29,  39,  86,  'Rep'),
    'Kentucky':       (8,  30,  25,  75,  'Rep'),
    'Louisiana':      (11, 27,  34,  68,  'Rep'),
    'Maine':          (22, 13,  79,  64,  'Dem'),
    'Maryland':       (32, 15,  99,  42,  'Dem'),
    'Massachusetts':  (37, 3,   126, 28,  'Dem'),
    'Michigan':       (16, 22,  53,  56,  'Rep'),
    'Minnesota':      (31, 34,  69,  64,  'Divided'),
    'Mississippi':    (16, 36,  43,  77,  'Rep'),
    'Missouri':       (10, 24,  48,  108, 'Rep'),
    'Montana':        (19, 31,  33,  67,  'Rep'),
    'Nebraska':       (None, None, None, None, 'Nonpartisan'),
    'Nevada':         (11, 9,   25,  16,  'Dem'),
    'New Hampshire':  (10, 13,  182, 206, 'Rep'),
    'New Jersey':     (24, 16,  46,  34,  'Dem'),
    'New Mexico':     (26, 15,  45,  24,  'Dem'),
    'New York':       (43, 20,  106, 43,  'Dem'),
    'North Carolina': (22, 28,  51,  69,  'Rep'),
    'North Dakota':   (7,  39,  14,  80,  'Rep'),
    'Ohio':           (8,  25,  35,  64,  'Rep'),
    'Oklahoma':       (9,  39,  18,  82,  'Rep'),
    'Pennsylvania':   (21, 28,  89,  113, 'Rep'),
    'Rhode Island':   (33, 5,   65,  10,  'Dem'),
    'South Carolina': (16, 30,  43,  81,  'Rep'),
    'South Dakota':   (3,  32,  8,   62,  'Rep'),
    'Tennessee':      (6,  27,  24,  73,  'Rep'),
    'Texas':          (13, 18,  65,  84,  'Rep'),
    'Utah':           (6,  23,  17,  58,  'Rep'),
    'Vermont':        (21, 7,   92,  46,  'Dem'),
    'Virginia':       (21, 19,  48,  52,  'Divided'),
    'Washington':     (29, 20,  57,  41,  'Dem'),
    'West Virginia':  (11, 23,  22,  78,  'Rep'),
    'Wisconsin':      (12, 21,  38,  58,  'Rep'),
    'Wyoming':        (2,  28,  7,   51,  'Rep'),
}

# ── Load psthomas CSV for 2015-2021 ──────────────────────────────────────────
raw = pd.read_csv(os.path.join(OUT_DIR, "state_legislature_raw.csv"))
raw = raw[raw['year'].between(2015, 2021)]
raw = raw[raw['state'].isin(STATES_49)].copy()

# Map leg_control from seat counts
def infer_leg_control(row):
    sd, sr = row.get('senate_dem', np.nan), row.get('senate_rep', np.nan)
    hd, hr = row.get('house_dem', np.nan), row.get('house_rep', np.nan)
    if pd.isna(sd) or pd.isna(sr) or pd.isna(hd) or pd.isna(hr):
        return 'unknown'
    # Nebraska unicameral nonpartisan
    if row['state'] == 'Nebraska':
        return 'Nonpartisan'
    sen_r = sr > sd
    hou_r = hr > hd
    sen_tie = sr == sd
    hou_tie = hr == hd
    if sen_r and hou_r:
        return 'Rep'
    if not sen_r and not hou_r and not sen_tie and not hou_tie:
        return 'Dem'
    return 'Split'

cols_keep = ['state', 'year', 'senate_dem', 'senate_rep', 'house_dem', 'house_rep']
hist = raw[cols_keep].copy()
hist['leg_control'] = raw.apply(infer_leg_control, axis=1)

# ── Build 2022 rows ───────────────────────────────────────────────────────────
rows_2022 = []
for state, (sd, sr, hd, hr, ctrl) in DATA_2022.items():
    rows_2022.append({
        'state': state, 'year': 2022,
        'senate_dem': sd, 'senate_rep': sr,
        'house_dem':  hd, 'house_rep':  hr,
        'leg_control': ctrl,
    })
yr22 = pd.DataFrame(rows_2022)

# ── Combine ───────────────────────────────────────────────────────────────────
legs = pd.concat([hist, yr22], ignore_index=True)
legs = legs.sort_values(['state', 'year']).reset_index(drop=True)

print(f"Total rows before governor merge: {len(legs)}")
print("States per year:")
print(legs.groupby('year')['state'].count().to_string())

# ── Derive pct and alignment variables ───────────────────────────────────────
def rep_pct(rep, dem):
    if pd.isna(rep) or pd.isna(dem):
        return np.nan
    total = rep + dem
    return rep / total if total > 0 else np.nan

legs['senate_rep_pct'] = legs.apply(
    lambda r: rep_pct(r['senate_rep'], r['senate_dem']), axis=1)
legs['house_rep_pct']  = legs.apply(
    lambda r: rep_pct(r['house_rep'],  r['house_dem']),  axis=1)
legs['unified_rep'] = (legs['leg_control'].str.startswith('Rep')).astype(float)
legs['unified_dem'] = (legs['leg_control'].str.startswith('Dem')).astype(float)

# ── Merge governor party ──────────────────────────────────────────────────────
gov = pd.read_csv(os.path.join(OUT_DIR, "governor_party.csv"))
gov['state'] = gov['state'].str.strip().str.title()
legs = legs.merge(gov[['state', 'year', 'republican_gov']], on=['state', 'year'], how='left')

def classify(row):
    g  = row['republican_gov']
    ur = row['unified_rep']
    ud = row['unified_dem']
    if pd.isna(g):
        return 'unknown'
    if g == 1 and ur == 1:
        return 'unified_rep'
    if g == 0 and ud == 1:
        return 'unified_dem'
    if g == 1 and ur != 1:
        return 'div_rep_gov'
    if g == 0 and ud != 1:
        return 'div_dem_gov'
    return 'mixed'

legs['gov_leg_alignment'] = legs.apply(classify, axis=1)
legs['align_unified_dem'] = (legs['gov_leg_alignment'] == 'unified_dem').astype(float)
legs['align_div_rep_gov'] = (legs['gov_leg_alignment'] == 'div_rep_gov').astype(float)
legs['align_div_dem_gov'] = (legs['gov_leg_alignment'] == 'div_dem_gov').astype(float)

# Drop gov column (already in governor_party.csv; avoid duplication in panel)
legs = legs.drop(columns=['republican_gov'])

out_path = os.path.join(OUT_DIR, "legislature_control.csv")
legs.to_csv(out_path, index=False)

print(f"\nSaved: {out_path}")
print(f"Total rows: {len(legs)}")
print(f"\nAlignment distribution (all years):")
print(legs['gov_leg_alignment'].value_counts().to_string())
print(f"\nSample (2022):")
print(legs[legs['year']==2022][['state','leg_control','gov_leg_alignment','senate_rep_pct','house_rep_pct']].to_string(index=False))
