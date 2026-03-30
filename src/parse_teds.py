"""
Parses SAMHSA TEDS-A (Treatment Episode Data Set - Admissions) 2006-2023.
Aggregates drug treatment admission counts by state and year (2015-2022).

ALCDRUG codes:
  0 = None
  1 = Alcohol only
  2 = Other drugs only
  3 = Alcohol and other drugs

We count ALL admissions (ALCDRUG 1, 2, 3) as "treatment admissions" for the
Treatment Admission Rate denominator, matching the project design doc.
A drug-only count (ALCDRUG 2 or 3) is also saved for reference.

STFIPS: standard Census/FIPS state codes (1=AL, 2=AK, 4=AZ, ...)
"""

import os
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
IN_FILE = os.path.join(BASE, "data", "raw", "teds", "tedsa_puf_2006_2023.csv")
OUT_DIR = os.path.join(BASE, "data", "raw", "teds")
os.makedirs(OUT_DIR, exist_ok=True)

# FIPS -> state name (standard Census codes)
FIPS_MAP = {
    1: 'Alabama',        2: 'Alaska',         4: 'Arizona',
    5: 'Arkansas',       6: 'California',      8: 'Colorado',
    9: 'Connecticut',   10: 'Delaware',       11: 'District Of Columbia',
   12: 'Florida',       13: 'Georgia',        15: 'Hawaii',
   16: 'Idaho',         17: 'Illinois',       18: 'Indiana',
   19: 'Iowa',          20: 'Kansas',         21: 'Kentucky',
   22: 'Louisiana',     23: 'Maine',          24: 'Maryland',
   25: 'Massachusetts', 26: 'Michigan',       27: 'Minnesota',
   28: 'Mississippi',   29: 'Missouri',       30: 'Montana',
   31: 'Nebraska',      32: 'Nevada',         33: 'New Hampshire',
   34: 'New Jersey',    35: 'New Mexico',     36: 'New York',
   37: 'North Carolina',38: 'North Dakota',   39: 'Ohio',
   40: 'Oklahoma',      41: 'Oregon',         42: 'Pennsylvania',
   44: 'Rhode Island',  45: 'South Carolina', 46: 'South Dakota',
   47: 'Tennessee',     48: 'Texas',          49: 'Utah',
   50: 'Vermont',       51: 'Virginia',       53: 'Washington',
   54: 'West Virginia', 55: 'Wisconsin',      56: 'Wyoming',
}

FIFTY_STATES = {k for k, v in FIPS_MAP.items() if v != 'District Of Columbia'}

YEARS = list(range(2015, 2023))

print(f"Reading {IN_FILE} ...")
print("(Loading only ADMYR, STFIPS, ALCDRUG columns)")

df = pd.read_csv(
    IN_FILE,
    usecols=['ADMYR', 'STFIPS', 'ALCDRUG'],
    dtype={'ADMYR': 'int16', 'STFIPS': 'int8', 'ALCDRUG': 'int8'},
)

print(f"Total rows loaded: {len(df):,}")

# Filter to 2015-2022 and 50 states
df = df[df['ADMYR'].isin(YEARS) & df['STFIPS'].isin(FIFTY_STATES)]
print(f"Rows after filtering to 2015-2022 + 50 states: {len(df):,}")

# All-substance treatment admissions (ALCDRUG 1, 2, 3 — exclude 0=none/missing)
df_valid = df[df['ALCDRUG'].isin([1, 2, 3])]

# Count total admissions and drug-only admissions per state-year
total_admissions = (
    df_valid.groupby(['STFIPS', 'ADMYR'])
    .size()
    .reset_index(name='treatment_admissions')
)

drug_only = (
    df_valid[df_valid['ALCDRUG'].isin([2, 3])]
    .groupby(['STFIPS', 'ADMYR'])
    .size()
    .reset_index(name='drug_treatment_admissions')
)

result = total_admissions.merge(drug_only, on=['STFIPS', 'ADMYR'], how='left')

# Map FIPS to state name
result['state'] = result['STFIPS'].map(FIPS_MAP)
result = result.rename(columns={'ADMYR': 'year'})
result = result[['state', 'year', 'treatment_admissions', 'drug_treatment_admissions']]
result = result.sort_values(['state', 'year']).reset_index(drop=True)

out_path = os.path.join(OUT_DIR, "teds_2015_2022_combined.csv")
result.to_csv(out_path, index=False)

print(f"\nSaved: {out_path}")
print(f"Total rows: {len(result)}")
print(f"States per year:")
print(result.groupby('year')['state'].count().to_string())
print(f"\nNational totals by year:")
nat = result.groupby('year')[['treatment_admissions', 'drug_treatment_admissions']].sum()
print(nat.to_string())
print("\nSample:")
print(result.head(10).to_string(index=False))
