"""
Parses FBI drug arrest data for all 50 states, 2015-2022.

Strategy:
  2015-2019: FBI ASR (Age, Sex, Race) 12-Month National Master Files
             Fixed-width ASCII, "1980-Current" record layout (LRECL=564)
  2020-2022: FBI CIUS Table 69 "Arrest by State" Excel files
             (ASR unreliable post-2020 due to NIBRS transition)

Output: data/raw/nibrs/nibrs_2015_2022_combined.csv
"""

import os
import re
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE, "data", "raw", "nibrs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── ASR numeric state code → full state name ──────────────────────────────────
STATE_MAP = {
    '01': 'ALABAMA',        '02': 'ARIZONA',         '03': 'ARKANSAS',
    '04': 'CALIFORNIA',     '05': 'COLORADO',         '06': 'CONNECTICUT',
    '07': 'DELAWARE',       '08': 'DISTRICT OF COLUMBIA', '09': 'FLORIDA',
    '10': 'GEORGIA',        '11': 'IDAHO',            '12': 'ILLINOIS',
    '13': 'INDIANA',        '14': 'IOWA',             '15': 'KANSAS',
    '16': 'KENTUCKY',       '17': 'LOUISIANA',        '18': 'MAINE',
    '19': 'MARYLAND',       '20': 'MASSACHUSETTS',    '21': 'MICHIGAN',
    '22': 'MINNESOTA',      '23': 'MISSISSIPPI',      '24': 'MISSOURI',
    '25': 'MONTANA',        '26': 'NEBRASKA',         '27': 'NEVADA',
    '28': 'NEW HAMPSHIRE',  '29': 'NEW JERSEY',       '30': 'NEW MEXICO',
    '31': 'NEW YORK',       '32': 'NORTH CAROLINA',   '33': 'NORTH DAKOTA',
    '34': 'OHIO',           '35': 'OKLAHOMA',         '36': 'OREGON',
    '37': 'PENNSYLVANIA',   '38': 'RHODE ISLAND',     '39': 'SOUTH CAROLINA',
    '40': 'SOUTH DAKOTA',   '41': 'TENNESSEE',        '42': 'TEXAS',
    '43': 'UTAH',           '44': 'VERMONT',          '45': 'VIRGINIA',
    '46': 'WASHINGTON',     '47': 'WEST VIRGINIA',    '48': 'WISCONSIN',
    '49': 'WYOMING',        '50': 'ALASKA',           '51': 'HAWAII',
}

# 50 states only (exclude DC)
TARGET_STATES = {k for k, v in STATE_MAP.items() if v != 'DISTRICT OF COLUMBIA'}


def total_arrests(line):
    """Sum all male + female age-group arrest counts from the ASR record."""
    total = 0
    # Male age groups: 1-indexed pos 41-238 = 0-indexed 40:238, 22 groups x 9 chars
    # Female age groups: 1-indexed pos 239-436 = 0-indexed 238:436, 22 groups x 9 chars
    for start in (40, 238):
        for i in range(22):
            s = line[start + i * 9: start + i * 9 + 9].strip()
            if s:
                try:
                    total += int(s)
                except ValueError:
                    pass
    return total


def parse_asr_year(filepath, year):
    """
    Parse one ASR 12-Month National Master File.
    Returns a list of dicts: {state, year, drug_arrests}.

    Drug offense codes:
      "18"  = total drug abuse violations (preferred)
      "180" = drug sale/manufacture subtotal
      "185" = drug possession subtotal
    Strategy: use "18" when an ORI reports it; otherwise use "180"+"185".
    """
    # state_code -> {ori -> {code -> count}}
    from collections import defaultdict
    state_ori_data = defaultdict(lambda: defaultdict(dict))

    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            if len(line) < 26:
                continue
            if line[0] != '3':          # record identifier must be '3'
                continue
            state_code = line[1:3]
            if state_code not in TARGET_STATES:
                continue
            offense_code = line[22:25].rstrip()
            if offense_code not in ('18', '180', '185'):
                continue
            ori = line[3:10]
            count = total_arrests(line)
            state_ori_data[state_code][ori][offense_code] = count

    rows = []
    for state_code, ori_dict in state_ori_data.items():
        state_total = 0
        for ori, codes in ori_dict.items():
            if '18' in codes:
                state_total += codes['18']
            else:
                state_total += codes.get('180', 0) + codes.get('185', 0)
        rows.append({
            'state': STATE_MAP[state_code].title(),
            'year': year,
            'drug_arrests': state_total,
            'source': 'ASR',
        })
    return rows


# ── ASR file paths for 2015-2019 ─────────────────────────────────────────────
ASR_FILES = {
    2015: os.path.join(BASE, "data/raw/nibrs/asr-2015/asr-2015/2015_ASR12MON_NATIONAL_MASTER_FILE.txt"),
    2016: os.path.join(BASE, "data/raw/nibrs/asr-2016/asr-2016/2016_ASR12MON_NATIONAL_MASTER_FILE.txt"),
    2017: os.path.join(BASE, "data/raw/nibrs/asr-2017/2017_ASR12MON_NATIONAL_MASTER_FILE.txt"),
    2018: os.path.join(BASE, "data/raw/nibrs/asr-2018/2018_ASR12MON_NATIONAL_MASTER_FILE.txt"),
    2019: os.path.join(BASE, "data/raw/nibrs/asr-2019/asr-2019/2019_ASR12MON_NATIONAL_MASTER_FILE.txt"),
}

# ── Table 69 Excel paths for 2020-2022 ───────────────────────────────────────
TABLE69_FILES = {
    2020: (os.path.join(BASE, "data/raw/nibrs/persons-arrested-2020/Table_69_Arrest_by_State_2020.xls"),  'xlrd'),
    2021: (os.path.join(BASE, "data/raw/nibrs/persons-arrested-2021/Table_69_Arrest_by_State_2021.xls"),  'xlrd'),
    2022: (os.path.join(BASE, "data/raw/nibrs/persons-arrested-2022/Table_69_Arrest_by_State_2022.xlsx"), 'openpyxl'),
}

DRUG_COL = 22   # column index of "Drug abuse violations" in Table 69


def clean_state_name(raw):
    """Strip footnote numbers, punctuation, and extra whitespace from Table 69 state names."""
    if not isinstance(raw, str):
        return None
    # Remove digits (footnote markers), commas, and extra whitespace
    cleaned = re.sub(r'[\d,]+', '', raw).strip()
    return cleaned.upper() if cleaned else None


def parse_table69_year(filepath, engine, year):
    """
    Parse one Table 69 Excel file.
    Returns a list of dicts: {state, year, drug_arrests}.
    """
    df = pd.read_excel(filepath, engine=engine, header=None)

    rows = []
    current_state = None

    for _, row in df.iterrows():
        col0 = row.iloc[0]
        col1 = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ''
        drug_val = row.iloc[DRUG_COL] if len(row) > DRUG_COL else None

        if isinstance(col0, str) and col0.strip() and col0.strip() not in (
            'State', 'Table 69', 'Arrests', f'by State, {year}'
        ):
            cleaned = clean_state_name(col0)
            if cleaned and not cleaned.startswith('NOTE') and not cleaned.startswith('DOES'):
                current_state = cleaned

        if 'Total all ages' in col1 and current_state:
            # Map to title-case for consistency
            state_title = current_state.title()
            # Only keep 50 states (exclude DC, territories)
            if current_state in [v for v in STATE_MAP.values() if v != 'DISTRICT OF COLUMBIA']:
                try:
                    drug_arrests = int(drug_val) if pd.notna(drug_val) else 0
                except (ValueError, TypeError):
                    drug_arrests = 0
                rows.append({
                    'state': state_title,
                    'year': year,
                    'drug_arrests': drug_arrests,
                    'source': 'Table69',
                })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
all_rows = []

print("=== Parsing ASR Master Files (2015-2019) ===")
for year, path in sorted(ASR_FILES.items()):
    print(f"  {year}: {os.path.basename(path)} ... ", end='', flush=True)
    rows = parse_asr_year(path, year)
    national = sum(r['drug_arrests'] for r in rows)
    print(f"{len(rows)} states, national total = {national:,}")
    all_rows.extend(rows)

print("\n=== Parsing Table 69 Excel Files (2020-2022) ===")
for year, (path, engine) in sorted(TABLE69_FILES.items()):
    print(f"  {year}: {os.path.basename(path)} ... ", end='', flush=True)
    rows = parse_table69_year(path, engine, year)
    national = sum(r['drug_arrests'] for r in rows)
    print(f"{len(rows)} states, national total = {national:,}")
    all_rows.extend(rows)

df = pd.DataFrame(all_rows)
df = df.sort_values(['state', 'year']).reset_index(drop=True)

# Save combined
combined_path = os.path.join(OUT_DIR, "nibrs_2015_2022_combined.csv")
df.to_csv(combined_path, index=False)

print(f"\nSaved: {combined_path}")
print(f"Total rows: {len(df)}")
print(f"States covered per year:")
print(df.groupby('year')['state'].count().to_string())
print("\nSample (first 10 rows):")
print(df.head(10).to_string(index=False))
