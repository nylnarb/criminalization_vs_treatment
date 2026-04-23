"""
Parses N-SSATS (2015-2020) and N-SUMHSS (2021-2022) facility-level survey data.
Aggregates to state-year level and outputs supply-side treatment infrastructure
variables for merging into panel_dataset.csv.

Source files expected in data/raw/:
  N-SSATS-2015-DS0001-data/N-SSATS-2015-DS0001-data-excel.csv
  NSSATSPUF_2016.csv
  NSSATS_PUF_2017_CSV.csv
  NSSATS_PUF_2018_CSV.csv
  NSSATS_PUF_2019_CSV.csv
  NSSATS_PUF_2020_CSV.csv
  NSUMHSS_2021_PUF_CSV.csv
  NSUMHSS_2022_PUF_CSV.csv

Output variables (per state-year):
  facilities_per_100k     — substance use treatment facilities per 100k residents
  public_facility_pct     — % of facilities that are government-operated (OWNERSHP 3-6)
  private_fp_pct          — % of facilities that are private for-profit (OWNERSHP 2)
  medicaid_pct            — % of facilities accepting Medicaid (PAYASST=1)
  beds_per_100k           — residential + hospital inpatient beds per 100k residents
  capacity_utilization    — mean occupancy % among facilities with beds (B_PCT)

OWNERSHP codes (consistent across all years):
  1 = Private nonprofit
  2 = Private for-profit
  3 = Local/county/community government
  4 = State government
  5 = Federal government (incl. VA)
  6 = Tribal government

FOCUS codes (N-SUMHSS 2021-2022 only — used to filter to substance use facilities):
  1 = Substance use treatment only
  2 = Mental health treatment only  (excluded)
  3 = Both substance use and mental health
  4 = General health / other        (excluded)
"""

import os
import numpy as np
import pandas as pd

BASE    = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE, "data", "raw")
OUT_DIR = os.path.join(BASE, "data", "raw", "nsumhss")
PANEL_PATH = os.path.join(BASE, "data", "processed", "panel_dataset.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ── File registry ──────────────────────────────────────────────────────────────
FILE_MAP = {
    2015: os.path.join(RAW_DIR, "N-SSATS-2015-DS0001-data", "N-SSATS-2015-DS0001-data-excel.csv"),
    2016: os.path.join(RAW_DIR, "NSSATSPUF_2016.csv"),
    2017: os.path.join(RAW_DIR, "NSSATS_PUF_2017_CSV.csv"),
    2018: os.path.join(RAW_DIR, "NSSATS_PUF_2018_CSV.csv"),
    2019: os.path.join(RAW_DIR, "NSSATS_PUF_2019_CSV.csv"),
    2020: os.path.join(RAW_DIR, "NSSATS_PUF_2020_CSV.csv"),
    2021: os.path.join(RAW_DIR, "NSUMHSS_2021_PUF_CSV.csv"),
    2022: os.path.join(RAW_DIR, "NSUMHSS_2022_PUF_CSV.csv"),
}

# ── State abbreviation → name (for 2015/2016/2021/2022) ───────────────────────
ABBR_TO_STATE = {
    'AL':'Alabama', 'AK':'Alaska', 'AZ':'Arizona', 'AR':'Arkansas',
    'CA':'California', 'CO':'Colorado', 'CT':'Connecticut', 'DE':'Delaware',
    'FL':'Florida', 'GA':'Georgia', 'HI':'Hawaii', 'ID':'Idaho',
    'IL':'Illinois', 'IN':'Indiana', 'IA':'Iowa', 'KS':'Kansas',
    'KY':'Kentucky', 'LA':'Louisiana', 'ME':'Maine', 'MD':'Maryland',
    'MA':'Massachusetts', 'MI':'Michigan', 'MN':'Minnesota', 'MS':'Mississippi',
    'MO':'Missouri', 'MT':'Montana', 'NE':'Nebraska', 'NV':'Nevada',
    'NH':'New Hampshire', 'NJ':'New Jersey', 'NM':'New Mexico', 'NY':'New York',
    'NC':'North Carolina', 'ND':'North Dakota', 'OH':'Ohio', 'OK':'Oklahoma',
    'OR':'Oregon', 'PA':'Pennsylvania', 'RI':'Rhode Island', 'SC':'South Carolina',
    'SD':'South Dakota', 'TN':'Tennessee', 'TX':'Texas', 'UT':'Utah',
    'VT':'Vermont', 'VA':'Virginia', 'WA':'Washington', 'WV':'West Virginia',
    'WI':'Wisconsin', 'WY':'Wyoming', 'DC':'District Of Columbia',
}

# ── FIPS → name (for 2017-2020) ───────────────────────────────────────────────
FIPS_TO_STATE = {
    1:'Alabama', 2:'Alaska', 4:'Arizona', 5:'Arkansas', 6:'California',
    8:'Colorado', 9:'Connecticut', 10:'Delaware', 11:'District Of Columbia',
    12:'Florida', 13:'Georgia', 15:'Hawaii', 16:'Idaho', 17:'Illinois',
    18:'Indiana', 19:'Iowa', 20:'Kansas', 21:'Kentucky', 22:'Louisiana',
    23:'Maine', 24:'Maryland', 25:'Massachusetts', 26:'Michigan', 27:'Minnesota',
    28:'Mississippi', 29:'Missouri', 30:'Montana', 31:'Nebraska', 32:'Nevada',
    33:'New Hampshire', 34:'New Jersey', 35:'New Mexico', 36:'New York',
    37:'North Carolina', 38:'North Dakota', 39:'Ohio', 40:'Oklahoma',
    41:'Oregon', 42:'Pennsylvania', 44:'Rhode Island', 45:'South Carolina',
    46:'South Dakota', 47:'Tennessee', 48:'Texas', 49:'Utah', 50:'Vermont',
    51:'Virginia', 53:'Washington', 54:'West Virginia', 55:'Wisconsin',
    56:'Wyoming',
}

FIFTY_STATES = set(FIPS_TO_STATE.values()) - {'District Of Columbia'}

def to_num(series):
    return pd.to_numeric(series, errors='coerce')

def load_year(year, path):
    df = pd.read_csv(path, low_memory=False)
    df['year'] = year

    # ── Resolve state column to state name ────────────────────────────────────
    if 'STFIPS' in df.columns:
        df['state'] = to_num(df['STFIPS']).map(FIPS_TO_STATE)
    elif 'STATE' in df.columns:
        df['state'] = df['STATE'].astype(str).str.strip().str.upper().map(ABBR_TO_STATE)
    elif 'LOCATIONSTATE' in df.columns:
        df['state'] = df['LOCATIONSTATE'].astype(str).str.strip().str.upper().map(ABBR_TO_STATE)
    else:
        raise ValueError(f"{year}: no STATE, STFIPS, or LOCATIONSTATE column found")

    # ── Filter to substance use treatment facilities ───────────────────────────
    if year >= 2021:
        # N-SUMHSS: keep FOCUS 1 (SU only) or 3 (SU + MH)
        focus = to_num(df['FOCUS'])
        df = df[focus.isin([1, 3])].copy()
    else:
        # N-SSATS: keep facilities offering substance use treatment
        if 'TREATMT' in df.columns:
            treatmt = to_num(df['TREATMT'])
            df = df[treatmt == 1].copy()

    # ── Filter to 50 states ───────────────────────────────────────────────────
    df = df[df['state'].isin(FIFTY_STATES)].copy()

    # ── Coerce key numeric columns ────────────────────────────────────────────
    df['OWNERSHP'] = to_num(df['OWNERSHP'])
    df['PAYASST']  = to_num(df['PAYASST'])
    df['HOSPBED']  = to_num(df.get('HOSPBED', pd.Series(dtype=float)))
    df['RESBED']   = to_num(df.get('RESBED',  pd.Series(dtype=float)))
    df['B_PCT']    = to_num(df.get('B_PCT',   pd.Series(dtype=float)))

    # Total beds per facility (residential + hospital inpatient)
    df['total_beds'] = df['HOSPBED'].fillna(0) + df['RESBED'].fillna(0)

    # Ownership flags
    df['is_public']      = df['OWNERSHP'].isin([3, 4, 5, 6]).astype(float)
    df['is_private_fp']  = (df['OWNERSHP'] == 2).astype(float)
    df['ownershp_valid'] = df['OWNERSHP'].notna().astype(float)

    # Medicaid flag
    df['medicaid_yes']   = (df['PAYASST'] == 1).astype(float)
    df['payasst_valid']  = df['PAYASST'].notna().astype(float)

    # Occupancy — only meaningful for facilities that have beds
    df['bpct_valid'] = ((df['B_PCT'].notna()) & (df['total_beds'] > 0)).astype(float)

    return df[['state','year','is_public','is_private_fp','ownershp_valid',
               'medicaid_yes','payasst_valid','total_beds','B_PCT','bpct_valid']]


# ── Load and aggregate all years ──────────────────────────────────────────────
print("Loading population from panel_dataset.csv...")
panel = pd.read_csv(PANEL_PATH)
panel['state'] = panel['state'].str.strip().str.title()
pop = panel[['state','year','population']].drop_duplicates().copy()

all_years = []
for year, path in FILE_MAP.items():
    print(f"  Processing {year}...")
    df = load_year(year, path)

    grp = df.groupby(['state','year'])
    agg = grp.agg(
        facility_count    =('is_public',      'count'),
        public_n          =('is_public',       'sum'),
        private_fp_n      =('is_private_fp',   'sum'),
        ownershp_d        =('ownershp_valid',  'sum'),
        medicaid_n        =('medicaid_yes',    'sum'),
        payasst_d         =('payasst_valid',   'sum'),
        total_beds        =('total_beds',      'sum'),
        bpct_sum          =('B_PCT',           lambda x: x[df.loc[x.index,'bpct_valid']==1].sum()),
        bpct_n            =('bpct_valid',      'sum'),
    ).reset_index()

    def safe_pct(num, denom):
        return np.where(denom > 0, num / denom * 100, np.nan)

    agg['public_facility_pct'] = safe_pct(agg['public_n'],     agg['ownershp_d'])
    agg['private_fp_pct']      = safe_pct(agg['private_fp_n'], agg['ownershp_d'])
    agg['medicaid_pct']        = safe_pct(agg['medicaid_n'],   agg['payasst_d'])
    agg['capacity_utilization']= safe_pct(agg['bpct_sum'],     agg['bpct_n'])

    all_years.append(agg)
    print(f"    {year}: {len(df):,} facilities → {agg['state'].nunique()} states")

result = pd.concat(all_years, ignore_index=True)

# ── Merge population and compute per-100k rates ───────────────────────────────
result = result.merge(pop, on=['state','year'], how='left')
result['facilities_per_100k'] = result['facility_count'] / result['population'] * 100_000
result['beds_per_100k']       = result['total_beds']     / result['population'] * 100_000

KEEP_COLS = ['state','year','facilities_per_100k','beds_per_100k',
             'public_facility_pct','private_fp_pct','medicaid_pct','capacity_utilization']
result = result[KEEP_COLS].sort_values(['state','year']).reset_index(drop=True)

out_path = os.path.join(OUT_DIR, "nsumhss_2015_2022.csv")
result.to_csv(out_path, index=False)

print(f"\nSaved: {out_path}")
print(f"Rows: {len(result)} | States per year: {result.groupby('year')['state'].count().to_dict()}")
print("\nNational averages:")
print(result[['facilities_per_100k','beds_per_100k','public_facility_pct',
              'private_fp_pct','medicaid_pct','capacity_utilization']].mean().round(2).to_string())
print("\nSample:")
print(result.head(10).to_string(index=False))
