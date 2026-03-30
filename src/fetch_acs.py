"""
Fetches Census ACS 5-Year estimates for all 50 states, 2015-2022.
Variables: population, poverty rate, median income, race/ethnicity, unemployment.
Saves one CSV per year to data/raw/acs/
No API key required for these requests.
"""

import requests
import pandas as pd
import os
import time

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "acs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = list(range(2015, 2023))

# ACS 5-Year variable definitions
# B01003_001E = total population
# B17001_002E = population below poverty level
# B17001_001E = total population for poverty calc
# B19013_001E = median household income
# B03002_003E = white alone (non-Hispanic)
# B03002_004E = Black alone (non-Hispanic)
# B03002_012E = Hispanic/Latino
# B23025_005E = unemployed (civilian labor force)
# B23025_002E = civilian labor force (denominator)
# B01003_001E already covers population

VARIABLES = {
    "B01003_001E": "population",
    "B17001_002E": "poverty_pop",
    "B17001_001E": "poverty_universe",
    "B19013_001E": "median_income",
    "B03002_003E": "pop_white_nonhisp",
    "B03002_004E": "pop_black_nonhisp",
    "B03002_012E": "pop_hispanic",
    "B23025_005E": "unemployed",
    "B23025_002E": "labor_force",
}

var_string = ",".join(VARIABLES.keys())

def fetch_year(year):
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": f"NAME,{var_string}",
        "for": "state:*",
    }
    print(f"  Fetching {year}...", end=" ")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cols = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=cols)

    # Rename raw variable codes to human-readable names
    df = df.rename(columns=VARIABLES)
    df["year"] = year

    # Convert numeric columns
    numeric_cols = list(VARIABLES.values()) + []
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive rates
    df["poverty_rate"] = df["poverty_pop"] / df["poverty_universe"]
    df["unemployment_rate"] = df["unemployed"] / df["labor_force"]
    df["pct_white"] = df["pop_white_nonhisp"] / df["population"]
    df["pct_black"] = df["pop_black_nonhisp"] / df["population"]
    df["pct_hispanic"] = df["pop_hispanic"] / df["population"]

    # Clean up: drop FIPS numeric state code, keep NAME as "state"
    df = df.drop(columns=["state"], errors="ignore")  # drop FIPS state code col
    df = df.rename(columns={"NAME": "state"})

    keep = [
        "state", "year", "population",
        "poverty_rate", "median_income",
        "unemployment_rate",
        "pct_white", "pct_black", "pct_hispanic",
        "pop_white_nonhisp", "pop_black_nonhisp", "pop_hispanic",
    ]
    # state col may already be renamed
    available = [c for c in keep if c in df.columns]
    df = df[available]

    out_path = os.path.join(OUTPUT_DIR, f"acs_{year}.csv")
    df.to_csv(out_path, index=False)
    print(f"saved {len(df)} rows -> {out_path}")
    return df

all_frames = []
for year in YEARS:
    try:
        df = fetch_year(year)
        all_frames.append(df)
        time.sleep(0.5)  # be polite to the API
    except Exception as e:
        print(f"  ERROR for {year}: {e}")

if all_frames:
    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, "acs_2015_2022_combined.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined ACS file saved: {combined_path}")
    print(f"Total rows: {len(combined)}")
    print(combined.head())
