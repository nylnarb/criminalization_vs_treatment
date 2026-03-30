"""
Fetches drug arrest counts by state and year (2015-2022) from the
FBI Crime Data Explorer public API.
Requires a free API key from https://api.data.gov/signup/
Usage: python fetch_nibrs.py --api-key YOUR_KEY_HERE
Saves one CSV per year + combined file to data/raw/nibrs/
"""

import requests
import pandas as pd
import os
import time
import argparse

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nibrs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = list(range(2015, 2023))

# All 50 state abbreviations
STATE_ABBRS = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

BASE_URL = "https://api.usa.gov/crime/fbi/cde"

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", required=True, help="Free key from https://api.data.gov/signup/")
args = parser.parse_args()
API_KEY = args.api_key

def fetch_state_drug_arrests(state_abbr):
    """
    Fetches drug offense arrest counts for a state across all years.
    Endpoint returns year-by-year data for the state.
    """
    url = f"{BASE_URL}/arrest/state/{state_abbr}/drug-offenses"
    params = {
        "from": 2015,
        "to": 2022,
        "API_KEY": API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()

all_rows = []
print("Fetching drug arrest data from FBI CDE API...")

for abbr in STATE_ABBRS:
    print(f"  {abbr}...", end=" ")
    try:
        data = fetch_state_drug_arrests(abbr)
        if data is None:
            print("no data")
            continue

        # Response structure varies — handle both list and dict formats
        if isinstance(data, dict) and "data" in data:
            records = data["data"]
        elif isinstance(data, list):
            records = data
        else:
            print(f"unexpected format: {type(data)}")
            continue

        for record in records:
            if isinstance(record, dict):
                year = record.get("year") or record.get("data_year")
                # Sum all drug offense subtypes if broken out
                if "actual" in record:
                    arrests = record["actual"]
                elif "Drug/Narcotic Offenses" in record:
                    arrests = record["Drug/Narcotic Offenses"]
                else:
                    # Try to sum numeric values
                    arrests = sum(v for v in record.values() if isinstance(v, (int, float)))

                if year and arrests is not None:
                    all_rows.append({
                        "state_abbr": abbr,
                        "year": int(year),
                        "drug_arrests": int(arrests) if arrests else 0,
                    })

        print(f"ok ({len(records)} records)")
        time.sleep(0.3)

    except Exception as e:
        print(f"ERROR: {e}")

if not all_rows:
    print("\nNo data retrieved from CDE API — endpoint may require API key.")
    print("See manual download instructions in the Implementation Guide.")
else:
    df = pd.DataFrame(all_rows)

    # Map abbreviations to full state names
    import us
    df["state"] = df["state_abbr"].apply(lambda a: us.states.lookup(a).name if us.states.lookup(a) else a)
    df = df[df["year"].between(2015, 2022)]
    df = df.sort_values(["state", "year"])

    # Save per-year files
    for year, group in df.groupby("year"):
        path = os.path.join(OUTPUT_DIR, f"nibrs_{year}.csv")
        group.to_csv(path, index=False)

    # Save combined
    combined_path = os.path.join(OUTPUT_DIR, "nibrs_2015_2022_combined.csv")
    df.to_csv(combined_path, index=False)

    print(f"\nNIBRS combined file saved: {combined_path}")
    print(f"Total rows: {len(df)}")
    print(df.head(10).to_string())
