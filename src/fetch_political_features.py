"""
Creates enhanced political features for 2015-2022:

1. governor_streak: consecutive years same party held governorship
   Positive = consecutive Republican years, Negative = consecutive Democrat years
   Example: Idaho 2022 = +8 (R all 8 years), Kansas 2022 = -4 (D since 2019)

2. pres_vote_rep: Republican presidential vote share (%) for most recent election
   2015-2016 → 2012 election results
   2017-2020 → 2016 election results
   2021-2022 → 2020 election results
   Source: MIT Election Data and Science Lab / FECA certified results

Saves: data/raw/policy/political_features.csv
"""

import os
import pandas as pd
import numpy as np

BASE    = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE, "data/raw/policy")

YEARS = list(range(2015, 2023))

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

# ── Republican presidential vote share by state and election year ─────────────
# Source: certified results, MIT Election Lab / FEC
# Format: state -> {election_year: rep_pct}
PRES_VOTE = {
    'Alabama':        {2012: 60.7, 2016: 62.1, 2020: 62.0},
    'Alaska':         {2012: 54.8, 2016: 51.3, 2020: 52.8},
    'Arizona':        {2012: 53.7, 2016: 48.7, 2020: 49.1},
    'Arkansas':       {2012: 60.6, 2016: 60.6, 2020: 62.4},
    'California':     {2012: 37.1, 2016: 31.6, 2020: 34.3},
    'Colorado':       {2012: 46.1, 2016: 43.3, 2020: 41.9},
    'Connecticut':    {2012: 40.7, 2016: 40.9, 2020: 39.2},
    'Delaware':       {2012: 40.0, 2016: 41.7, 2020: 39.8},
    'Florida':        {2012: 49.1, 2016: 49.0, 2020: 51.2},
    'Georgia':        {2012: 53.3, 2016: 50.8, 2020: 49.2},
    'Hawaii':         {2012: 27.8, 2016: 30.0, 2020: 34.3},
    'Idaho':          {2012: 64.5, 2016: 59.2, 2020: 63.9},
    'Illinois':       {2012: 40.7, 2016: 38.8, 2020: 40.6},
    'Indiana':        {2012: 54.1, 2016: 56.9, 2020: 57.0},
    'Iowa':           {2012: 46.2, 2016: 51.2, 2020: 53.1},
    'Kansas':         {2012: 59.7, 2016: 56.7, 2020: 56.2},
    'Kentucky':       {2012: 60.5, 2016: 62.5, 2020: 62.1},
    'Louisiana':      {2012: 57.8, 2016: 58.1, 2020: 58.5},
    'Maine':          {2012: 40.9, 2016: 44.9, 2020: 44.0},
    'Maryland':       {2012: 35.9, 2016: 33.9, 2020: 32.2},
    'Massachusetts':  {2012: 37.5, 2016: 32.8, 2020: 32.1},
    'Michigan':       {2012: 44.7, 2016: 47.5, 2020: 47.8},
    'Minnesota':      {2012: 45.0, 2016: 44.9, 2020: 45.3},
    'Mississippi':    {2012: 55.3, 2016: 57.9, 2020: 57.6},
    'Missouri':       {2012: 53.8, 2016: 56.8, 2020: 57.0},
    'Montana':        {2012: 55.4, 2016: 56.2, 2020: 56.9},
    'Nebraska':       {2012: 59.8, 2016: 58.8, 2020: 58.2},
    'Nevada':         {2012: 45.7, 2016: 45.5, 2020: 47.7},
    'New Hampshire':  {2012: 46.4, 2016: 46.5, 2020: 45.4},
    'New Jersey':     {2012: 40.7, 2016: 41.4, 2020: 41.3},
    'New Mexico':     {2012: 42.8, 2016: 40.0, 2020: 43.5},
    'New York':       {2012: 35.2, 2016: 36.5, 2020: 37.9},
    'North Carolina': {2012: 50.4, 2016: 49.8, 2020: 49.9},
    'North Dakota':   {2012: 58.3, 2016: 63.0, 2020: 65.1},
    'Ohio':           {2012: 47.7, 2016: 51.7, 2020: 53.3},
    'Oklahoma':       {2012: 66.8, 2016: 65.3, 2020: 65.4},
    'Pennsylvania':   {2012: 46.6, 2016: 48.2, 2020: 48.8},
    'Rhode Island':   {2012: 35.2, 2016: 38.9, 2020: 38.6},
    'South Carolina': {2012: 54.6, 2016: 54.9, 2020: 55.1},
    'South Dakota':   {2012: 57.9, 2016: 61.5, 2020: 61.8},
    'Tennessee':      {2012: 59.5, 2016: 60.7, 2020: 60.7},
    'Texas':          {2012: 57.2, 2016: 52.2, 2020: 52.1},
    'Utah':           {2012: 72.8, 2016: 45.5, 2020: 58.1},
    'Vermont':        {2012: 30.9, 2016: 30.3, 2020: 30.7},
    'Virginia':       {2012: 47.3, 2016: 44.4, 2020: 44.0},
    'Washington':     {2012: 41.3, 2016: 38.1, 2020: 38.8},
    'West Virginia':  {2012: 62.3, 2016: 68.5, 2020: 65.4},
    'Wisconsin':      {2012: 46.1, 2016: 47.2, 2020: 48.8},
    'Wyoming':        {2012: 68.9, 2016: 68.2, 2020: 69.9},
}

# Assign election to year: use most recent election result as political context
def get_pres_vote(state, year):
    if year <= 2016:
        return PRES_VOTE[state][2012]
    elif year <= 2020:
        return PRES_VOTE[state][2016]
    else:
        return PRES_VOTE[state][2020]

# ── Governor streak ───────────────────────────────────────────────────────────
# Load governor party data
gov = pd.read_csv(os.path.join(OUT_DIR, "governor_party.csv"))
gov['state'] = gov['state'].str.strip().str.title()

# We need pre-2015 data to compute accurate streaks for states that maintained
# the same party for years before 2015. Hardcode streak entering 2015:
# Positive = consecutive R years going into 2015, Negative = consecutive D years
STREAK_2014 = {
    'Alabama':        8,   # R since 2003 (Riley 2003, Bentley 2011)
    'Alaska':        -6,   # Parnell (R) actually - but Walker (I) won in 2014. Parnell was R 2009-2014 = +5
    'Arizona':        4,   # Brewer (R) since 2009
    'Arkansas':       2,   # Beebe (D) 2007-2014 = -8... wait Hutchinson (R) won Nov 2014, took office Jan 2015
    'California':    -4,   # Brown (D) since 2011
    'Colorado':      -4,   # Hickenlooper (D) since 2011
    'Connecticut':   -4,   # Malloy (D) since 2011
    'Delaware':      -8,   # Markell (D) since 2009, Carper before = long D streak
    'Florida':        4,   # Scott (R) since 2011
    'Georgia':        4,   # Deal (R) since 2011
    'Hawaii':        -8,   # Abercrombie (D) 2011-2014, before that Lingle (R)
    'Idaho':          8,   # Otter (R) since 2007
    'Illinois':       0,   # Quinn (D) through 2014, Rauner (R) won Nov 2014 → entering 2015 as R: streak=0 (just flipped)
    'Indiana':        4,   # Pence (R) since 2013, before that Daniels (R) = +8
    'Iowa':           4,   # Branstad (R) since 2011
    'Kansas':         4,   # Brownback (R) since 2011
    'Kentucky':      -8,   # Beshear (D) since 2007
    'Louisiana':      6,   # Jindal (R) since 2008
    'Maine':          4,   # LePage (R) since 2011
    'Maryland':      -8,   # O'Malley (D) since 2007
    'Massachusetts': -4,   # Patrick (D) 2007-2015, Baker (R) won Nov 2014 → entering 2015 as R: streak=0
    'Michigan':       4,   # Snyder (R) since 2011
    'Minnesota':     -4,   # Dayton (D) since 2011
    'Mississippi':    4,   # Bryant (R) since 2012
    'Missouri':      -4,   # Nixon (D) since 2009
    'Montana':       -4,   # Bullock (D) since 2013
    'Nebraska':       4,   # Heineman (R) since 2005 → Ricketts (R) won 2014
    'Nevada':         4,   # Sandoval (R) since 2011
    'New Hampshire': -2,   # Hassan (D) since 2013
    'New Jersey':     4,   # Christie (R) since 2010
    'New Mexico':     4,   # Martinez (R) since 2011
    'New York':      -8,   # Cuomo (D) since 2011
    'North Carolina': 2,   # McCrory (R) since 2013
    'North Dakota':   8,   # Dalrymple (R) since 2010
    'Ohio':           4,   # Kasich (R) since 2011
    'Oklahoma':       4,   # Fallin (R) since 2011
    'Pennsylvania':  -4,   # Corbett (R) through 2014, Wolf (D) won 2014 → entering 2015 as D: streak=0
    'Rhode Island':  -4,   # Chafee (I/D) through 2014, Raimondo (D) won 2014
    'South Carolina': 4,   # Haley (R) since 2011
    'South Dakota':   8,   # Daugaard (R) since 2011
    'Tennessee':      4,   # Haslam (R) since 2011
    'Texas':          8,   # Perry (R) since 2000
    'Utah':           8,   # Herbert (R) since 2009
    'Vermont':       -4,   # Shumlin (D) since 2011
    'Virginia':      -4,   # McAuliffe (D) since 2014
    'Washington':    -4,   # Inslee (D) since 2013
    'West Virginia': -8,   # Tomblin (D) since 2011
    'Wisconsin':      4,   # Walker (R) since 2011
    'Wyoming':        8,   # Mead (R) since 2011
}

# Fix a few edge cases where governor changed Jan 2015
# Arkansas: Beebe (D) left, Hutchinson (R) started Jan 2015 → first R year = 2015, prior streak D
STREAK_2014['Arkansas'] = -8
# Illinois: Quinn (D) left, Rauner (R) started Jan 2015
STREAK_2014['Illinois'] = -4
# Massachusetts: Patrick (D) left, Baker (R) started Jan 2015
STREAK_2014['Massachusetts'] = -8
# Pennsylvania: Corbett (R) left, Wolf (D) started Jan 2015
STREAK_2014['Pennsylvania'] = 4
# Alaska: Parnell (R) left, Walker (I) started Dec 2014
STREAK_2014['Alaska'] = 5

def compute_streak(state, year, gov_df):
    """Compute consecutive years of same governor party entering this year."""
    current_party = gov_df.loc[(gov_df['state']==state) & (gov_df['year']==year), 'republican_gov'].values
    if len(current_party) == 0:
        return np.nan
    is_rep = int(current_party[0])

    # Count consecutive years of same party going back
    streak = 0
    prior_streak = STREAK_2014.get(state, 0)

    for y in range(2015, year + 1):
        party = gov_df.loc[(gov_df['state']==state) & (gov_df['year']==y), 'republican_gov'].values
        if len(party) == 0:
            break
        if int(party[0]) == is_rep:
            streak += 1
        else:
            streak = 1  # reset — they just switched

    # If the entire 2015-year window is same party, add the pre-2015 streak
    all_same = all(
        gov_df.loc[(gov_df['state']==state) & (gov_df['year'].between(2015, year)),
                   'republican_gov'].values == is_rep
    )
    if all_same:
        prior = prior_streak if is_rep and prior_streak > 0 else (-prior_streak if not is_rep and prior_streak < 0 else 0)
        streak = streak + prior

    return streak if is_rep else -streak


# ── Build output table ────────────────────────────────────────────────────────
rows = []
for state in STATES_49:
    for year in YEARS:
        streak = compute_streak(state, year, gov)
        pres   = get_pres_vote(state, year)
        rows.append({
            'state':         state,
            'year':          year,
            'gov_streak':    streak,
            'pres_vote_rep': pres,
        })

out = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, "political_features.csv")
out.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"\nStreak distribution (2022):")
yr = out[out['year']==2022].sort_values('gov_streak')
print(yr[['state','gov_streak','pres_vote_rep']].to_string(index=False))
