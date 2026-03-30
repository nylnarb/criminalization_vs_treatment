"""
Creates two policy feature tables for 2015-2022:
  1. recreational_marijuana.csv  -- was recreational marijuana legal in this state-year?
  2. governor_party.csv          -- was the governor Republican in this state-year?

Sources:
  Marijuana: NCSL state marijuana laws database (dates are well-documented public record)
  Governor:  National Governors Association historical records

Coding:
  marijuana_legal: 1 = recreational possession legal at any point during the year
  republican_gov:  1 = Republican governor, 0 = Democrat/Independent
"""

import os
import pandas as pd

BASE    = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE, "data", "raw", "policy")
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = list(range(2015, 2023))

# ── Recreational Marijuana Legalization ───────────────────────────────────────
# First year in which recreational possession became LEGAL in each state.
# Source: NCSL, Ballotpedia (these dates are unambiguous public record)
# Oregon excluded from analysis entirely.
MJ_LEGAL_FROM = {
    'Colorado':      2013,  # Amendment 64, Jan 2013
    'Washington':    2013,  # I-502, Dec 2012/Jan 2013
    'Alaska':        2015,  # Measure 2, Feb 2015
    'California':    2017,  # Prop 64, Nov 2016; possession legal Jan 2017
    'Nevada':        2017,  # Question 2, Jan 2017
    'Massachusetts': 2017,  # Question 4, Dec 2016; retail Jan 2017
    'Maine':         2017,  # Question 1, Jan 2017 (retail delayed to 2020)
    'Vermont':       2018,  # Act 86, Jul 2018 (possession only; sales 2022)
    'Michigan':      2019,  # Prop 1, Dec 2018; effective Dec 2019
    'Illinois':      2020,  # HB 1438, Jan 2020
    'Arizona':       2021,  # Prop 207, Nov 2020; effective Jan 2021
    'Montana':       2021,  # CI-118/I-190, Jan 2021
    'New Jersey':    2021,  # Public Question 1, Feb 2021
    'New Mexico':    2021,  # HB 2, Jun 2021
    'Connecticut':   2021,  # SB 1302, Jul 2021
    'New York':      2021,  # MRTA, Mar 2021
    'Rhode Island':  2022,  # SB 2430, May 2022
    'Missouri':      2022,  # Amendment 3, Dec 2022
}
# All other states in our dataset: 0 throughout 2015-2022

STATES_49 = [
    'Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut',
    'Delaware','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa',
    'Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan',
    'Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada',
    'New Hampshire','New Jersey','New Mexico','New York','North Carolina',
    'North Dakota','Ohio','Oklahoma','Pennsylvania','Rhode Island',
    'South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont',
    'Virginia','Washington','West Virginia','Wisconsin','Wyoming',
]

mj_rows = []
for state in STATES_49:
    legal_from = MJ_LEGAL_FROM.get(state, 9999)
    for year in YEARS:
        mj_rows.append({
            'state': state,
            'year': year,
            'marijuana_legal': 1 if year >= legal_from else 0,
        })

mj_df = pd.DataFrame(mj_rows)
mj_path = os.path.join(OUT_DIR, "recreational_marijuana.csv")
mj_df.to_csv(mj_path, index=False)
print(f"Saved: {mj_path}")
print(f"  States with legal marijuana by year:")
print(mj_df.groupby('year')['marijuana_legal'].sum().to_string())

# ── Governor's Party ──────────────────────────────────────────────────────────
# 1 = Republican governor, 0 = Democrat or Independent
# Source: National Governors Association, Ballotpedia gubernatorial history
# Each entry is the party of the governor serving the MAJORITY of that calendar year.
# Notable edge cases:
#   Alaska 2015-2018: Bill Walker (Independent) → coded 0
#   West Virginia 2017: Jim Justice elected as D, switched to R Aug 2017 → coded 0 (D majority of year)
#   West Virginia 2018-2022: Jim Justice (R after switch) → coded 1
#   Louisiana 2015: Bobby Jindal (R) through Jan 2016 → 1
#   Louisiana 2016-2022: John Bel Edwards (D) → 0
#   Kentucky 2015: Bevin (R) took office Dec 2015, Beshear (D) all prior → coded 0
#   Kentucky 2016-2019: Matt Bevin (R) → 1
#   Kentucky 2020-2022: Andy Beshear (D) → 0

GOV_PARTY = {
    # state: {year: is_republican}
    'Alabama':        {y: 1 for y in YEARS},
    'Alaska':         {2015:0, 2016:0, 2017:0, 2018:0, 2019:1, 2020:1, 2021:1, 2022:1},
    'Arizona':        {y: 1 for y in YEARS},
    'Arkansas':       {y: 1 for y in YEARS},
    'California':     {y: 0 for y in YEARS},
    'Colorado':       {y: 0 for y in YEARS},
    'Connecticut':    {y: 0 for y in YEARS},
    'Delaware':       {y: 0 for y in YEARS},
    'Florida':        {y: 1 for y in YEARS},
    'Georgia':        {y: 1 for y in YEARS},
    'Hawaii':         {y: 0 for y in YEARS},
    'Idaho':          {y: 1 for y in YEARS},
    'Illinois':       {2015:1, 2016:1, 2017:1, 2018:1, 2019:0, 2020:0, 2021:0, 2022:0},
    'Indiana':        {y: 1 for y in YEARS},
    'Iowa':           {y: 1 for y in YEARS},
    'Kansas':         {2015:1, 2016:1, 2017:1, 2018:1, 2019:0, 2020:0, 2021:0, 2022:0},
    'Kentucky':       {2015:0, 2016:1, 2017:1, 2018:1, 2019:1, 2020:0, 2021:0, 2022:0},
    'Louisiana':      {2015:1, 2016:0, 2017:0, 2018:0, 2019:0, 2020:0, 2021:0, 2022:0},
    'Maine':          {2015:1, 2016:1, 2017:1, 2018:1, 2019:0, 2020:0, 2021:0, 2022:0},
    'Maryland':       {y: 1 for y in YEARS},  # Larry Hogan (R) 2015-2023
    'Massachusetts':  {y: 1 for y in YEARS},  # Charlie Baker (R) 2015-2023
    'Michigan':       {2015:1, 2016:1, 2017:1, 2018:1, 2019:0, 2020:0, 2021:0, 2022:0},
    'Minnesota':      {y: 0 for y in YEARS},
    'Mississippi':    {y: 1 for y in YEARS},
    'Missouri':       {2015:0, 2016:0, 2017:1, 2018:1, 2019:1, 2020:1, 2021:1, 2022:1},
    'Montana':        {2015:0, 2016:0, 2017:0, 2018:0, 2019:0, 2020:0, 2021:1, 2022:1},
    'Nebraska':       {y: 1 for y in YEARS},
    'Nevada':         {2015:1, 2016:1, 2017:1, 2018:1, 2019:0, 2020:0, 2021:0, 2022:0},
    'New Hampshire':  {2015:0, 2016:0, 2017:1, 2018:1, 2019:1, 2020:1, 2021:1, 2022:1},
    'New Jersey':     {2015:1, 2016:1, 2017:1, 2018:0, 2019:0, 2020:0, 2021:0, 2022:0},
    'New Mexico':     {2015:1, 2016:1, 2017:1, 2018:1, 2019:0, 2020:0, 2021:0, 2022:0},
    'New York':       {y: 0 for y in YEARS},
    'North Carolina': {2015:1, 2016:1, 2017:0, 2018:0, 2019:0, 2020:0, 2021:0, 2022:0},
    'North Dakota':   {y: 1 for y in YEARS},
    'Ohio':           {y: 1 for y in YEARS},
    'Oklahoma':       {y: 1 for y in YEARS},
    'Pennsylvania':   {2015:0, 2016:0, 2017:0, 2018:0, 2019:0, 2020:0, 2021:0, 2022:0},
    'Rhode Island':   {y: 0 for y in YEARS},
    'South Carolina': {y: 1 for y in YEARS},
    'South Dakota':   {y: 1 for y in YEARS},
    'Tennessee':      {y: 1 for y in YEARS},
    'Texas':          {y: 1 for y in YEARS},
    'Utah':           {y: 1 for y in YEARS},
    'Vermont':        {2015:0, 2016:0, 2017:1, 2018:1, 2019:1, 2020:1, 2021:1, 2022:1},
    'Virginia':       {2015:0, 2016:0, 2017:0, 2018:0, 2019:0, 2020:0, 2021:0, 2022:1},
    'Washington':     {y: 0 for y in YEARS},
    'West Virginia':  {2015:0, 2016:0, 2017:0, 2018:1, 2019:1, 2020:1, 2021:1, 2022:1},
    'Wisconsin':      {2015:1, 2016:1, 2017:1, 2018:1, 2019:0, 2020:0, 2021:0, 2022:0},
    'Wyoming':        {y: 1 for y in YEARS},
}

gov_rows = []
for state in STATES_49:
    for year in YEARS:
        gov_rows.append({
            'state': state,
            'year': year,
            'republican_gov': GOV_PARTY[state][year],
        })

gov_df = pd.DataFrame(gov_rows)
gov_path = os.path.join(OUT_DIR, "governor_party.csv")
gov_df.to_csv(gov_path, index=False)
print(f"\nSaved: {gov_path}")
print(f"  Republican governors by year:")
print(gov_df.groupby('year')['republican_gov'].sum().to_string())
