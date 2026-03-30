Criminalization Vs. Treatment

The Problem: 
Drugs can be treated as a criminal issue or a public health issue. 
Response by states vary wildly
Arrests and treatment admissions may reflect structural policy choices

The Question: 
Does state demographic and socioeconomic structure explain variation in arrest-to-treatment ratios?

Data Sources (to be merged) : 
FBI UCR / NIBRS → Drug arrest counts
SAMHSA TEDS → Treatment admissions
U.S. Census ACS → Demographic & socioeconomic variables

https://www.fbi.gov/how-we-can-help-you/more-fbi-services-and-information/ucr/nibrs
https://www.samhsa.gov/data/data-we-collect/teds-treatment-episode-data-set/datafiles/teds-a-2019
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
Demographics
Poverty rate
Median income

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

Geographic region indicators (Northeast, Midwest, West) are included as model features and rank among the strongest predictors of the Criminalization Index. This is a model limitation, not a finding. Region captures unmeasured structural variation — likely including treatment infrastructure capacity, harm reduction policy culture, and urban density — that is not represented in the current feature set. High regional importance signals that relevant variables are absent from the model rather than that geography itself causes variation in criminalization. Future work should incorporate treatment bed capacity per capita, drug court availability, and state behavioral health spending, which are not available in consistent series across all states and years in this study period.







