Criminalization Vs. Treatment

The Problem: 
Drugs can be treated as a criminal issue or a public health issue. 
Response by states vary wildly
Arrests and treatment admissions may reflect structural policy choices

The Question: 
Does state demographic and socioeconomic structure explain variation in arrest-to-treatment ratios?

Data Sources: 
FBI UCR / NIBRS → Drug arrest counts
SAMHSA TEDS-A (Treatment Episode Data Set – Admissions) → Treatment admissions and treatment system characteristics
SAMHSA N-SSATS (2015–2020) / N-SUMHSS (2021–2022) → Facility-level supply: bed capacity, ownership type, Medicaid acceptance
U.S. Census ACS → Demographic & socioeconomic variables
CDC WONDER → Drug overdose death counts
Bureau of Justice Statistics / Vera Institute → State incarceration rates
State legislative records → Governor party, legislature partisan composition, marijuana legalization

https://www.fbi.gov/how-we-can-help-you/more-fbi-services-and-information/ucr/nibrs
https://www.samhsa.gov/data/data-we-collect/teds-treatment-episode-data-set/datafiles/teds-a-2019
https://www.samhsa.gov/data/data-we-collect/n-ssats-national-survey-substance-abuse-treatment-services
https://www.census.gov/programs-surveys/acs.html

Scope:
What: State-Level Enforcement & Treatment Metrics
When: Multi-Year Panel (2015-2022)
Where: All 50 states

Data Set Construction: 
Each Observation represents a State-Year (e.g., North Carolina - 2018)

Variables Created:
Drug Arrest Rate (per 100,000 residents)
Treatment Admission Rate (per 100,000 residents)
Criminalization Index = Arrest Rate / Treatment Rate
Demographics: poverty rate, median income, unemployment rate, racial composition
Political/policy: governor party, presidential vote share, governor streak, legislature partisan control, marijuana legalization status
Structural: incarceration rate, overdose death rate

Treatment system characteristics (from TEDS-A, per state-year):
  CJ referral % — share of admissions referred by courts or criminal justice
  MAT adoption % — share of admissions receiving medication-assisted treatment
  Opioid case mix % — share of admissions involving heroin, fentanyl, or other opioids
  Meth case mix % — share of admissions involving methamphetamine or amphetamines
  Dual diagnosis % — share of admissions with co-occurring psychiatric condition
  Same-day access % — share of admissions with no wait for treatment
  Residential % — share of admissions in inpatient or residential settings
  Repeat treatment % — share of admissions with at least one prior episode

Facility supply variables (from N-SSATS/N-SUMHSS, per state-year):
  Facilities per 100k — licensed treatment centers per 100,000 residents
  Beds per 100k — residential and inpatient beds per 100,000 residents
  Public facility % — share of facilities that are government-operated
  Private for-profit % — share of facilities that are private for-profit
  Medicaid acceptance % — share of facilities accepting Medicaid
  Capacity utilization — average bed occupancy among facilities with residential beds

Target Variable: Criminalization Index = Arrest Rate / Treatment Rate

Data Processing Steps:
Standardized state names
Converted counts to population-adjusted rates
Handled missing NIBRS transition years
Normalized metrics for cross-state comparison

Method:
Dependent Variable: Criminalization Index
Independent Variables:
	Demographic (poverty rate, median income, unemployment rate, % white, % black, % hispanic)
	Socio-Economic
	Structural factors (incarceration rate, overdose death rate, marijuana legalization status, governor's party)

Considered but excluded:
	Police officers per capita (FBI LEE data) — excluded due to inconsistent agency-level reporting causing implausible year-to-year state swings (CV up to 38% for some states). Nebraska also missing entirely. May be revisited with a cleaner source.

Possible Models: 
Linear Regression
Random Forest
Gradient Boosting

What are we doing beyond just comparing numbers?
 Predictive Modeling
Model variation in Criminalization Index across states and years
Estimate which variables most strongly predict enforcement vs treatment orientation
 Multivariate Analysis
Control for demographic and socioeconomic factors
Isolate structural predictors (e.g., incarceration rate, treatment capacity)
Comparative Testing
Evaluate model performance (regression vs machine learning models)
Compare explanatory strength of structural vs demographic variables
Policy-Relevant Interpretation
Identify which factors are most associated with higher criminalization
Explore implications for state-level policy orientation

Visualization Strategy: 
U.S. Choropleth Map 
	Objective: Assess geographic variation in state-level policy orientation.
	Display Criminalization Index by state
	Identify geographic clustering patterns

Time Trend Line Graph
	Objective: Examine temporal shifts in enforcement versus treatment orientation.
		Compare arrest and treatment rates over time
 		Identify divergence or convergence trends across years

Feature Importance Plot
	Objective: Evaluate which predictors most strongly influence variation in the Criminalization Index.
	 Rank model predictors by importance
	 Compare structural versus demographic explanatory power

Scatter Plot with Regression Line
	Objective: Assess the relationship between structural factors and criminalization intensity.
	 Visualize predictor-to-index relationships
	Display model fit and residual variation

Why This Matters?
	Puts real numbers behind the enforcement vs. treatment debate.
	Shows whether policy is actually shifting — or just being talked about.
	Tracks how priorities change over time, not just year to year.

What We Add?
	Moves the conversation beyond headlines and opinions.
	Turns abstract policy language into measurable trends.
	Visually shows divergence or convergence between arrest and treatment rates.

Why It’s Useful?
Helps policymakers see where resources are truly going.
Clarifies whether public health goals match enforcement practices.
Provides a grounded starting point for future reform discussions.

Limitations:
Oregon is excluded from the panel analysis. Oregon is the only state to have fully decriminalized drug possession during this period (Measure 110, effective February 2021) and the only state that has never participated in SAMHSA TEDS — using its own state-level OHA data infrastructure instead. No publicly available treatment admissions series comparable to TEDS exists for Oregon across 2015-2022. Its subsequent legislative reversal (HB 4002, 2024), which recriminalized possession amid rising overdose deaths and slow treatment infrastructure deployment, suggests the absence of transparent federal reporting may have compounded difficulty in evaluating the policy in real time. Oregon’s exclusion is a meaningful gap, not merely a technical one.

Idaho and Florida are excluded entirely from all analysis and visualizations due to unresolvable data-quality issues. Idaho progressively withdrew from SAMHSA TEDS reporting between 2018 and 2022 — submitted treatment admissions fell from ~2,860/year to 278, while arrest rates remained flat, causing the Criminalization Index to inflate from 3.6 to 46.9 by 2022. This is a reporting artifact, not a real policy signal. Florida never submitted ASR data to FBI UCR for 2015–2016, has unreliable values for 2017–2021 due to early NIBRS transition, and only has one usable year (2022) — insufficient for meaningful cross-state comparison. Including either state would distort state averages and model estimates.

Illinois is retained but years 2020–2021 are excluded. NIBRS transition artifacts reduce Illinois arrest counts to near-zero in those years (arrest rate 2.2–2.9/100k vs. 84–187 in surrounding years), artificially inverting the index. The remaining six years (2015–2019, 2022) are consistent and comparable to other states. Several other states (Alabama, Maryland, New Jersey, New Mexico, New York, Pennsylvania) also have reduced reporting in 2020–2021 due to the NIBRS transition; affected state-years are excluded from modeling where noted.

Geographic region indicators (Northeast, Midwest, West) are included as model features and rank among the strongest predictors of the Criminalization Index. This is a model limitation, not a finding. Region captures unmeasured structural variation — treatment infrastructure capacity, harm reduction policy culture, and urban density — that was not represented in the original feature set. The current version addresses this partially by incorporating N-SSATS/N-SUMHSS facility supply variables (beds per 100k, facilities per 100k, ownership mix, Medicaid acceptance, capacity utilization) and TEDS-A treatment system characteristics (MAT adoption, CJ referral share, opioid/meth case mix). If region importance declines in model runs with these variables included, it confirms that infrastructure and treatment system composition were driving the regional signal. Remaining regional variation likely reflects drug court availability, state behavioral health spending, harm reduction policy culture, and urban density patterns not captured in the current dataset.

Key Findings:

National arrest rates have been declining faster than treatment rates since 2016. Population-weighted arrest rates peaked at 437.9 per 100k in 2016 and fell to 188.2 by 2021 — a 57% drop — before partially recovering to 227.7 in 2022. Treatment admission rates also declined, from 505.1 per 100k in 2018 to 306.6 in 2022, but more slowly. The net effect is that the national population-weighted criminalization index has been trending downward (approximately 0.88 in 2015 to 0.74 in 2022), with a sharp dip in 2020–2021 driven by COVID-19 disruptions to both arrest activity and treatment access. Year is not a statistically significant predictor in OLS (p=0.62), indicating that state-level structural variation far exceeds any national time trend.

Governor party is not a meaningful predictor; presidential vote share is. Republican governor (`republican_gov`) is uncorrelated with the criminalization index once other factors are controlled (p=0.88, coef +0.011). Presidential vote share (`pres_vote_rep`) is highly significant (p=0.008, coef +0.84). States that vote more Republican in presidential elections have consistently higher indexes, but this holds regardless of who currently holds the governorship. Criminalization orientation appears to reflect entrenched political culture, not the current administration.

Poverty and overdose death rates both negatively correlate with the index — contrary to intuition. Higher poverty (p=0.003, coef −0.84) and higher overdose death rates (p=0.003, coef −0.92) are each independently associated with lower criminalization. For overdose deaths, the likely mechanism is that states hit hardest by the opioid crisis (West Virginia, Ohio, Pennsylvania) substantially expanded treatment capacity during this period, growing the treatment rate denominator faster than arrests grew. For poverty, the finding likely reflects that the most treatment-oriented states (Connecticut, Massachusetts, Vermont) are also among the wealthiest, with deep behavioral health infrastructure accumulated over decades. High poverty in isolation does not produce high criminalization.

The combination of poverty and incarceration amplifies criminalization more than either factor alone. The poverty × incarceration interaction (`incarc_x_poverty`) is the most statistically significant interaction term in OLS (p<0.001, coef +1.92) and the third-ranked feature in the Random Forest model (importance 0.110). Neither high poverty nor high incarceration alone is sufficient to produce extreme scores. States with both — Louisiana (avg index 3.62) and West Virginia (avg index 3.80) — have the most extreme profiles outside Idaho. The interaction suggests a structural trap: high incarceration infrastructure coexists with low treatment capacity in states that also face high poverty.

The opioid crisis produced divergent policy responses by political context. The presidential vote × overdose death rate interaction (`pres_x_overdose`) is statistically significant (p=0.001, coef +1.07). In Republican-voting states, higher overdose death rates are associated with higher criminalization. In Democratic-voting states, higher overdose death rates are associated with lower criminalization. The same public health crisis produced opposite policy orientations depending on the state's political environment. This is one of the clearest interaction effects in the data.

The relationship between predictors and criminalization is substantially non-linear. OLS achieves an R² of 0.41 while the best tree model (tuned XGBoost) achieves 0.68 — a 27-point gap. No single variable drives the index in isolation; the dominant predictors are interaction terms (`pres_x_incarceration`, `incarc_x_poverty`) rather than any individual feature. This indicates that criminalization orientation is produced by combinations of structural and political conditions rather than any single root cause.

Percent Black population negatively correlates with the criminalization index (p=0.018, coef −0.26). States with larger African American populations tend toward lower criminalization. The probable mechanism is that states with large urban centers — which correlate with larger African American populations — have higher absolute treatment admission rates due to greater treatment infrastructure density. This variable should be interpreted carefully: it is a demographic proxy for urbanization and treatment infrastructure, not a direct causal relationship.

Some states criminalize far more — or far less — than anything about them would predict. After accounting for every demographic, political, and structural variable in the model, certain states still land far from where they should. New Hampshire is the biggest outlier in the criminalizing direction: its poverty rate, political lean, incarceration rate, and regional profile all suggest it should be moderate, but it consistently arrests at a much higher rate relative to treatment than any of those factors explain. On the other side, Connecticut, Montana, and Vermont are all more treatment-oriented than predicted. Florida also lands well below its predicted score, though this is partly a data artifact given its incomplete reporting history. These residuals point to state-specific factors — local court diversion programs, treatment bed capacity, prosecution norms — that are not captured anywhere in the current dataset and represent the clearest direction for future data collection.

The poverty finding resolves when states are split by political lean. Across the full dataset, higher poverty correlates with lower criminalization — a result that seemed backwards. When states are divided into Republican-leaning and Democrat-leaning groups and analyzed separately, both groups show poverty increasing criminalization, not decreasing it. In Democrat-leaning states the effect has a slope of +7.23; in Republican-leaning states the slope is steeper at +10.76. The reason the overall correlation pointed the wrong direction is that wealthy, Democrat-leaning states like Connecticut and Massachusetts happen to also be the most treatment-oriented — so when you lump everyone together, the data looks like wealth causes treatment-orientation. It does not. Within each political group, more poverty means more criminalization. The difference is that Democrat-leaning states stay below the arrest-equals-treatment line even at high poverty, while Republican-leaning states cross above it. Poverty is a pressure that both groups feel, but only one group converts it into higher criminalization.







