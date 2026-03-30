"""
Parses FBI Law Enforcement Employees (LEE) agency-level data.
Aggregates to state-year level: total officers and officers per 100k population.

Uses ACS population for rate calculation (more accurate than summing agency coverage populations).

Saves: data/raw/lee/lee_state_2015_2022.csv
"""

import os
import pandas as pd
import us

BASE    = os.path.join(os.path.dirname(__file__), "..")
IN_FILE = os.path.join(BASE, "data/raw/lee/lee_1960_2024.csv")
ACS     = os.path.join(BASE, "data/raw/acs/acs_2015_2022_combined.csv")
OUT_DIR = os.path.join(BASE, "data/raw/lee")

FIFTY_STATES = {
    'Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut',
    'Delaware','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa',
    'Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan',
    'Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada',
    'New Hampshire','New Jersey','New Mexico','New York','North Carolina',
    'North Dakota','Ohio','Oklahoma','Pennsylvania','Rhode Island',
    'South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont',
    'Virginia','Washington','West Virginia','Wisconsin','Wyoming',
}

print("Loading LEE data...")
df = pd.read_csv(IN_FILE, usecols=['data_year','state_abbr','officer_ct','total_pe_ct'])
df = df[df['data_year'].between(2015, 2022)]

# Map abbreviation to full state name
df['state'] = df['state_abbr'].apply(
    lambda a: us.states.lookup(a).name if us.states.lookup(a) else None)
df = df[df['state'].isin(FIFTY_STATES)]

# Aggregate to state-year
agg = (df.groupby(['state', 'data_year'])
       .agg(total_officers=('officer_ct', 'sum'))
       .reset_index()
       .rename(columns={'data_year': 'year'}))

# Merge ACS population for rate calculation
acs = pd.read_csv(ACS, usecols=['state','year','population'])
acs['state'] = acs['state'].str.strip().str.title()
agg = agg.merge(acs, on=['state','year'], how='left')

agg['police_per_100k'] = agg['total_officers'] / agg['population'] * 100_000
agg = agg[['state','year','total_officers','police_per_100k']]
agg = agg.sort_values(['state','year']).reset_index(drop=True)

out_path = os.path.join(OUT_DIR, "lee_state_2015_2022.csv")
agg.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"Total rows: {len(agg)}")
print(f"States per year:")
print(agg.groupby('year')['state'].count().to_string())
print(f"\nMissing police_per_100k: {agg['police_per_100k'].isna().sum()}")
print(f"\nSample:")
print(agg.head(10).to_string(index=False))
