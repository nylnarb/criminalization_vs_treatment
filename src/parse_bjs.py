"""
Parses Vera Institute incarceration_trends_state.csv (sourced from BJS NPS).
Extracts state prison population rate per 100k for 2015-2022.

Saves: data/raw/bjs/bjs_incarceration_2015_2022.csv
"""

import os
import pandas as pd

BASE    = os.path.join(os.path.dirname(__file__), "..")
IN_FILE = os.path.join(BASE, "data/raw/bjs/incarceration_trends_state.csv")
OUT_DIR = os.path.join(BASE, "data/raw/bjs")

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

df = pd.read_csv(IN_FILE)
df = df[df['year'].between(2015, 2022)]
df = df[df['state_name'].isin(FIFTY_STATES)]
df = df[['state_name', 'year', 'total_prison_pop', 'total_prison_pop_rate']].copy()
df = df.rename(columns={
    'state_name':           'state',
    'total_prison_pop':     'prison_population',
    'total_prison_pop_rate':'incarceration_rate',
})
df = df.sort_values(['state','year']).reset_index(drop=True)

out_path = os.path.join(OUT_DIR, "bjs_incarceration_2015_2022.csv")
df.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"Total rows: {len(df)}")
print(f"States per year:")
print(df.groupby('year')['state'].count().to_string())
print(f"\nMissing incarceration_rate values: {df['incarceration_rate'].isna().sum()}")
print(f"\nSample:")
print(df.head(10).to_string(index=False))
