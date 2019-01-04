# Likelihood if policy holder will file claim 

In the automobile industry, a common question is how likely a policy-holder will file a claim during the coverage period. The task is to run several models and recommend which model runs better. Will only be using verified predictors which are: 

Target Variable:
•	CLAIM_FLAG: Claim Indicator (1 = Claim Filed, 0 = Otherwise) and 1 is the event value.

Nominal Predictor:
•	CREDIT_SCORE_BAND: Credit Score Tier (‘450 – 619’, ‘620 – 659’, ‘660 – 749’, and ‘750 +’)

Interval Predictors:
•	BLUEBOOK_1000: Blue Book Value in Thousands of Dollars (min. = 1.5, max. = 39.54)
•	CUST_LOYALTY: Number of Years with Company Before Policy Date (min. = 0, max. ≈ 21)
•	MVR_PTS: Motor Vehicle Record Points (min. = 0, max. = 10)
•	TIF: Time-in-Force (min. = 101, max. = 107)
•	TRAVTIME: Number of Miles Distance Commute to Work (min. = 5, max. ≈ 93) 

The models used are Nearest Neighbors model, Classification Tree model and the Logistic Regression model. 
