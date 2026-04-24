"""
Augments the existing panel_dataset.csv with new columns from:
  - TEDS-A (parse_teds.py output): treatment system characteristics
  - N-SSATS/N-SUMHSS (parse_nsumhss.py output): facility supply variables

Run this instead of 02_data_processing.py when the original raw source
files (NIBRS, ACS, BJS, etc.) are not available locally. The existing
panel_dataset.csv is used as the base and new columns are merged in.

Run order:
  1. python src/parse_teds.py
  2. python src/parse_nsumhss.py
  3. python src/02b_augment_panel.py
"""

import os
import pandas as pd

BASE      = os.path.join(os.path.dirname(__file__), "..")
PANEL     = os.path.join(BASE, "data", "processed", "panel_dataset.csv")
TEDS      = os.path.join(BASE, "data", "raw", "teds", "teds_2015_2022_combined.csv")
NSUMHSS   = os.path.join(BASE, "data", "raw", "nsumhss", "nsumhss_2015_2022.csv")

print("Loading panel_dataset.csv...")
panel = pd.read_csv(PANEL)
print(f"  {len(panel)} rows, {len(panel.columns)} columns")

# ── Merge TEDS-A treatment system variables ────────────────────────────────────
TEDS_NEW_COLS = [
    'cj_referral_pct', 'mat_pct', 'dual_dx_pct', 'same_day_pct',
    'opioid_pct', 'meth_pct', 'residential_pct', 'repeat_tx_pct',
]

if os.path.exists(TEDS):
    print("\nMerging TEDS-A variables...")
    teds = pd.read_csv(TEDS)
    teds['state'] = teds['state'].str.strip().str.title()

    # Drop columns that already exist in panel to avoid duplicates
    existing = [c for c in TEDS_NEW_COLS if c in panel.columns]
    if existing:
        print(f"  Replacing existing columns: {existing}")
        panel = panel.drop(columns=existing)

    merge_cols = ['state', 'year'] + [c for c in TEDS_NEW_COLS if c in teds.columns]
    panel = panel.merge(teds[merge_cols], on=['state', 'year'], how='left')

    added = [c for c in TEDS_NEW_COLS if c in panel.columns]
    print(f"  Added: {added}")
    print(f"  Coverage: {panel['cj_referral_pct'].notna().sum()} / {len(panel)} rows have CJ referral data")
else:
    print(f"  TEDS file not found — skipping ({TEDS})")

# ── Merge N-SSATS/N-SUMHSS facility supply variables ──────────────────────────
NSUMHSS_COLS = [
    'facilities_per_100k', 'beds_per_100k', 'public_facility_pct',
    'private_fp_pct', 'medicaid_pct', 'capacity_utilization',
]

if os.path.exists(NSUMHSS):
    print("\nMerging N-SSATS/N-SUMHSS variables...")
    nsumhss = pd.read_csv(NSUMHSS)
    nsumhss['state'] = nsumhss['state'].str.strip().str.title()

    existing = [c for c in NSUMHSS_COLS if c in panel.columns]
    if existing:
        print(f"  Replacing existing columns: {existing}")
        panel = panel.drop(columns=existing)

    merge_cols = ['state', 'year'] + [c for c in NSUMHSS_COLS if c in nsumhss.columns]
    panel = panel.merge(nsumhss[merge_cols], on=['state', 'year'], how='left')

    added = [c for c in NSUMHSS_COLS if c in panel.columns]
    print(f"  Added: {added}")
    print(f"  Coverage: {panel['beds_per_100k'].notna().sum()} / {len(panel)} rows have bed data")
else:
    print(f"  N-SUMHSS file not found — skipping ({NSUMHSS})")

# ── Save ───────────────────────────────────────────────────────────────────────
panel = panel.sort_values(['state', 'year']).reset_index(drop=True)
panel.to_csv(PANEL, index=False)

print(f"\nSaved: {PANEL}")
print(f"Final shape: {len(panel)} rows × {len(panel.columns)} columns")
print(f"Columns: {panel.columns.tolist()}")

print("\nMissing value counts for new columns:")
new_cols = [c for c in TEDS_NEW_COLS + NSUMHSS_COLS if c in panel.columns]
print(panel[new_cols].isna().sum().to_string())
