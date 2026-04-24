"""
Parses SAMHSA TEDS-A (Treatment Episode Data Set - Admissions) 2006-2023.
Aggregates drug treatment admissions and treatment characteristics by state
and year (2015-2022).

ALCDRUG codes:
  0 = None
  1 = Alcohol only
  2 = Other drugs only
  3 = Alcohol and other drugs

We count ALL admissions (ALCDRUG 1, 2, 3) as "treatment admissions" for the
Treatment Admission Rate denominator, matching the project design doc.
A drug-only count (ALCDRUG 2 or 3) is also saved for reference.

New variables extracted per state-year:
  cj_referral_pct   — % of admissions referred by court/criminal justice or DUI/DWI
                       (PSOURCE 6=court, 7=DUI/DWI; excludes -9 missing)
  mat_pct           — % of admissions receiving medication-assisted treatment
                       (METHUSE 1=yes; excludes -9 missing)
  opioid_pct        — % of admissions where heroin or other opiates/synthetics
                       is the primary substance (HERFLG | OPSYNFLG | METHFLG)
  meth_pct          — % of admissions where methamphetamine/amphetamines is
                       the primary substance (MTHAMFLG)
  dual_dx_pct       — % of admissions with co-occurring psychiatric problem
                       (PSYPROB 1=yes; excludes -9 missing)
  same_day_pct      — % of admissions with same-day access (DAYWAIT 0;
                       excludes -9 missing)
  residential_pct   — % of admissions in inpatient/residential settings
                       (SERVICES 1-2=detox inpatient/residential,
                        4-6=rehab residential/hospital)
  repeat_tx_pct     — % of admissions with at least one prior treatment episode
                       (NOPRIOR > 0; excludes -9 missing)

STFIPS: standard Census/FIPS state codes (1=AL, 2=AK, 4=AZ, ...)
"""

import os
import numpy as np
import pandas as pd

BASE    = os.path.join(os.path.dirname(__file__), "..")
IN_FILE = os.path.join(BASE, "data", "raw", "teds", "tedsa_puf_2006_2023.csv")
OUT_DIR = os.path.join(BASE, "data", "raw", "teds")
os.makedirs(OUT_DIR, exist_ok=True)

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
YEARS        = list(range(2015, 2023))

LOAD_COLS = [
    'ADMYR', 'STFIPS', 'ALCDRUG',
    'PSOURCE',   # referral source (6=court, 7=DUI/DWI)
    'METHUSE',   # medication-assisted treatment (1=yes)
    'PSYPROB',   # co-occurring psychiatric problem (1=yes)
    'DAYWAIT',   # days waiting for treatment (0=same day)
    'NOPRIOR',   # number of prior treatment episodes
    'HERFLG',    # heroin primary substance flag
    'OPSYNFLG',  # other opiates/synthetics primary substance flag
    'METHFLG',   # non-prescription methadone primary substance flag
    'MTHAMFLG',  # methamphetamine/amphetamines primary substance flag
    'SERVICES',  # service type (1-2=detox inpatient/residential, 4-6=rehab residential)
]

LOAD_DTYPES = {
    'ADMYR':   'int16',
    'STFIPS':  'int8',
    'ALCDRUG': 'int8',
    'PSOURCE': 'int8',
    'METHUSE': 'int8',
    'PSYPROB': 'int8',
    'DAYWAIT': 'int8',
    'NOPRIOR': 'int8',
    'HERFLG':  'int8',
    'OPSYNFLG':'int8',
    'METHFLG': 'int8',
    'MTHAMFLG':'int8',
    'SERVICES':'int8',
}

print(f"Reading {IN_FILE} ...")
print(f"(Loading {len(LOAD_COLS)} columns)")

df = pd.read_csv(IN_FILE, usecols=LOAD_COLS, dtype=LOAD_DTYPES)

print(f"Total rows loaded: {len(df):,}")

df = df[df['ADMYR'].isin(YEARS) & df['STFIPS'].isin(FIFTY_STATES)]
print(f"Rows after filtering to 2015-2022 + 50 states: {len(df):,}")

# ── Admission counts (existing logic) ─────────────────────────────────────────
df_valid = df[df['ALCDRUG'].isin([1, 2, 3])]

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

# ── Build binary indicator columns for vectorized groupby ─────────────────────
# For variables with -9 missing: track numerator (flag) and denominator (valid)
# separately so the rate excludes unknowns.

df['cj_flag']          = df['PSOURCE'].isin([6, 7]).astype('int8')
df['psource_valid']    = (df['PSOURCE'] != -9).astype('int8')

df['mat_flag']         = (df['METHUSE'] == 1).astype('int8')
df['methuse_valid']    = (df['METHUSE'] != -9).astype('int8')

df['dualdx_flag']      = (df['PSYPROB'] == 1).astype('int8')
df['psyprob_valid']    = (df['PSYPROB'] != -9).astype('int8')

df['sameday_flag']     = (df['DAYWAIT'] == 0).astype('int8')
df['daywait_valid']    = (df['DAYWAIT'] >= 0).astype('int8')

df['repeat_flag']      = (df['NOPRIOR'] > 0).astype('int8')
df['noprior_valid']    = (df['NOPRIOR'] >= 0).astype('int8')

# Substance flags are 0/1 with no missing — use total row count as denominator
df['opioid_flag']      = ((df['HERFLG'] == 1) | (df['OPSYNFLG'] == 1) | (df['METHFLG'] == 1)).astype('int8')
df['meth_flag']        = (df['MTHAMFLG'] == 1).astype('int8')
df['residential_flag'] = df['SERVICES'].isin([1, 2, 4, 5, 6]).astype('int8')
df['total_flag']       = 1  # denominator for always-valid flags

# ── Aggregate to state-year ────────────────────────────────────────────────────
print("Aggregating to state-year...")

sums = df.groupby(['STFIPS', 'ADMYR']).agg(
    cj_n          =('cj_flag',          'sum'),
    cj_d          =('psource_valid',     'sum'),
    mat_n         =('mat_flag',          'sum'),
    mat_d         =('methuse_valid',     'sum'),
    dualdx_n      =('dualdx_flag',       'sum'),
    dualdx_d      =('psyprob_valid',     'sum'),
    sameday_n     =('sameday_flag',      'sum'),
    sameday_d     =('daywait_valid',     'sum'),
    opioid_n      =('opioid_flag',       'sum'),
    meth_n        =('meth_flag',         'sum'),
    residential_n =('residential_flag',  'sum'),
    repeat_n      =('repeat_flag',       'sum'),
    repeat_d      =('noprior_valid',     'sum'),
    total         =('total_flag',        'sum'),
).reset_index()

def safe_pct(num, denom):
    return np.where(denom > 0, num / denom * 100, np.nan)

sums['cj_referral_pct']  = safe_pct(sums['cj_n'],          sums['cj_d'])
sums['mat_pct']          = safe_pct(sums['mat_n'],         sums['mat_d'])
sums['dual_dx_pct']      = safe_pct(sums['dualdx_n'],      sums['dualdx_d'])
sums['same_day_pct']     = safe_pct(sums['sameday_n'],     sums['sameday_d'])
sums['opioid_pct']       = safe_pct(sums['opioid_n'],      sums['total'])
sums['meth_pct']         = safe_pct(sums['meth_n'],        sums['total'])
sums['residential_pct']  = safe_pct(sums['residential_n'], sums['total'])
sums['repeat_tx_pct']    = safe_pct(sums['repeat_n'],      sums['repeat_d'])

rate_cols = ['cj_referral_pct','mat_pct','dual_dx_pct','same_day_pct',
             'opioid_pct','meth_pct','residential_pct','repeat_tx_pct']

sums = sums[['STFIPS','ADMYR'] + rate_cols]

# ── Merge counts + rates ───────────────────────────────────────────────────────
result = result.merge(sums, on=['STFIPS', 'ADMYR'], how='left')

result['state'] = result['STFIPS'].map(FIPS_MAP)
result = result.rename(columns={'ADMYR': 'year'})
result = result[['state','year','treatment_admissions','drug_treatment_admissions'] + rate_cols]
result = result.sort_values(['state','year']).reset_index(drop=True)

out_path = os.path.join(OUT_DIR, "teds_2015_2022_combined.csv")
result.to_csv(out_path, index=False)

print(f"\nSaved: {out_path}")
print(f"Total rows: {len(result)}")
print(f"States per year:")
print(result.groupby('year')['state'].count().to_string())

print(f"\nNational totals by year:")
nat = result.groupby('year')[['treatment_admissions','drug_treatment_admissions']].sum()
print(nat.to_string())

print(f"\nNational averages for new rate variables:")
print(result[rate_cols].mean().round(1).to_string())

print("\nSample:")
print(result.head(10).to_string(index=False))
