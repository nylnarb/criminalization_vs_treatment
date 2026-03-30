"""
Fetches drug overdose death counts by state and year (2015-2022)
from CDC data.cdc.gov — VSRR Provisional Drug Overdose Death Counts
Dataset: xkb8-kh2a

Uses "12 month-ending December" for each year as a full-year proxy.
Indicator: "Number of Deaths" (all drug overdoses)

Saves: data/raw/cdc/cdc_overdose_2015_2022.csv
"""

import os
import requests
import pandas as pd

BASE    = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE, "data", "raw", "cdc")
os.makedirs(OUT_DIR, exist_ok=True)

DATASET = "xkb8-kh2a"
URL     = f"https://data.cdc.gov/resource/{DATASET}.json"

print("Fetching CDC overdose death counts (2015-2022)...")

resp = requests.get(
    URL,
    params={
        "$where":  "indicator='Number of Drug Overdose Deaths' AND period='12 month-ending' AND month='December'",
        "$select": "state, state_name, year, data_value, percent_complete, footnote_symbol",
        "$limit":  5000,
        "$order":  "year, state",
    },
    timeout=30,
)
resp.raise_for_status()
rows = resp.json()
print(f"  Rows returned: {len(rows)}")

df = pd.DataFrame(rows)
df["year"]           = df["year"].astype(int)
df["overdose_deaths"] = pd.to_numeric(df["data_value"], errors="coerce")

# Filter to 2015-2022 and 50 states (drop US total, NYC, territories)
EXCLUDE = {"US", "YC", "PR", "VI", "GU", "AS", "MP"}
df = df[df["year"].between(2015, 2022)]
df = df[~df["state"].isin(EXCLUDE)]

# Rename for consistency
df = df.rename(columns={"state_name": "state_full"})
df = df[["state_full", "state", "year", "overdose_deaths", "percent_complete", "footnote_symbol"]]
df = df.rename(columns={"state_full": "state", "state": "state_abbr"})

out_path = os.path.join(OUT_DIR, "cdc_overdose_2015_2022.csv")
df.to_csv(out_path, index=False)

print(f"\nSaved: {out_path}")
print(f"Total rows: {len(df)}")
print(f"States per year:")
print(df.groupby("year")["state"].count().to_string())
print(f"\nNational overdose deaths by year:")
print(df.groupby("year")["overdose_deaths"].sum().to_string())
print(f"\nSample:")
print(df.head(10).to_string(index=False))
