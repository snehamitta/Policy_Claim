import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import zero_one_loss
import statsmodels.api as api
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('/Users/snehamitta/Desktop/ML/MidTerm/policy_2001.csv')
print('No. of observations = ', data.shape[0])
print(data.groupby('CLAIM_FLAG').size()/data.shape[0])

data_train, data_test = train_test_split(data, test_size = 0.3, random_state = 20181010, stratify = data['CLAIM_FLAG'])

# #Q2.a) Number of observations in train and test set

print('Number of Observations in Training = ', data_train.shape[0])
print('Number of Observations in Testing = ', data_test.shape[0])

# #2.b) The claim rates in the train and testing portions

train = data_train.groupby('CLAIM_FLAG').size() / data_train.shape[0]
print('Distribution of CLAIM_FLAG = 1 in train set:', round(train.iloc[1], 6))
test = data_test.groupby('CLAIM_FLAG').size() / data_test.shape[0]
print('Distribution of CLAIM_FLAG = 1 in test set:', round(test.iloc[1], 6))

p1 = data_train[['CREDIT_SCORE_BAND']].astype('category')
X = pd.get_dummies(p1)
X_train = X.join(data_train[['BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME']])

p2 = data_test[['CREDIT_SCORE_BAND']].astype('category')
X = pd.get_dummies(p2)
X_test = X.join(data_test[['BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME']])

y_train = data_train['CLAIM_FLAG']
y_test = data_test['CLAIM_FLAG']

#Q2.c) 

# for model KNN 

knn = KNeighborsClassifier(n_neighbors=3, algorithm = 'brute', metric = 'euclidean')
knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)
print(y_pred1[0])
y_pred1_prob = knn.predict_proba(X_test)[:,1]

#Lift calculation for kNN
y_test_predProb = pd.DataFrame(knn.predict_proba(X_test))
y_test_predProb1 = y_test_predProb.set_index(y_test.index.values)
score_test_knn = pd.concat([y_test, y_test_predProb1], axis = 1)

# The AUC metric for kNN
fpr1, tpr1, thresholds = metrics.roc_curve(y_test, y_pred1_prob)
print('The AUC is for kNN model is', metrics.auc(fpr1, tpr1))

#RASE calculation
r1 = []
t1 = []
cats = data_test['CLAIM_FLAG'].unique()
for i in range (len(y_test)):
    s1 = []
    for cat in cats:
        if(y_pred1[i] == cat):
            s1.append((1-y_test_predProb.loc[i,cat])**2)
        else:
            s1.append((0-y_test_predProb.loc[i,cat])**2)
    r1.append(sum(s1))
t1 = sum(r1)
rase1 = sqrt(t1/(2*(len(y_test))))
print('The RASE value for kNN is', rase1)

#The RMS metric for kNN
rms = sqrt(mean_squared_error(y_test, y_pred1_prob))
print('The RMS value for kNN model is', rms)

# Checking of threshold for events
e = []
for i in range(len(y_pred1_prob)):
	if y_pred1_prob[i] >= train.iloc[1]:
		e.append(1)
	else:
		e.append(0)

# Misclassification rate for kNN
error_rate = zero_one_loss(y_test, e)
print('The misclassification rate of the kNN model is', error_rate)

# # for model Decision Tree 

dtc = DecisionTreeClassifier(criterion= 'entropy', max_depth = 10, random_state= 20181010)
dtc.fit(X_train, y_train)
y_pred2 = dtc.predict(X_test)
y_pred2_prob = dtc.predict_proba(X_test)[:,1]

#Lift calculation for Decision Tree
y_test_predProb = pd.DataFrame(dtc.predict_proba(X_test))
y_test_predProb2 = y_test_predProb.set_index(y_test.index.values)
score_test_dt = pd.concat([y_test, y_test_predProb2], axis = 1)

#The AUC metric for Decision Tree
fpr2, tpr2, thresholds = metrics.roc_curve(y_test, y_pred2_prob)
print('The AUC is for Decision Tree model is', metrics.auc(fpr2, tpr2))

#RASE calculation
r2 = []
t2 = []
cats = data_test['CLAIM_FLAG'].unique()
for i in range (len(y_test)):
    s2 = []
    for cat in cats:
        if(y_pred2[i] == cat):
            s2.append((1-y_test_predProb.loc[i,cat])**2)
        else:
            s2.append((0-y_test_predProb.loc[i,cat])**2)
    r2.append(sum(s2))
t2 = sum(r2)
rase2 = sqrt(t2/(2*(len(y_test))))
print('The RASE value for Decision Tree is', rase2)

#The RMS metric for decision tree
rms = sqrt(mean_squared_error(y_test, y_pred2_prob))
print('The RMS value for Decision Tree model is', rms)

#Checking for event with threshold
g = []
for i in range(len(y_pred2_prob)):
	if y_pred2_prob[i] >= train.iloc[1]:
		g.append(1)
	else:
		g.append(0)

#Misclassification Rate for Decision Tree
error_rate = zero_one_loss(y_test, g)
print('The misclassification rate of the Decision Tree model is', error_rate)

# for model Logistic Regression

logit = api.MNLogit(y_train, X_train)
print("Name of Target Variable:", logit.endog_names) 
print("Name(s) of Predictors:", logit.exog_names)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8) 
thisParameter = thisFit.params
print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_pred3 = thisFit.predict(X_test)
y_predict = list(pd.to_numeric(y_pred3.idxmax(axis=1)))
print(sum(y_predict))
y_pred3_prob = y_pred3.iloc[:,1]

#Lift calculation for Logistic Regression
y_test_predProb = thisFit.predict(X_test)
score_test_lr = pd.concat([y_test, y_test_predProb], axis = 1)

#The AUC metric for Logistic Regression
fpr3, tpr3, thresholds = metrics.roc_curve(y_test, y_pred3_prob)
print('The AUC is for Logistic model is', metrics.auc(fpr3, tpr3))

#RASE calculation
r3 = []
t3 = []
cats = data_test['CLAIM_FLAG'].unique()
for i in range (len(y_test)):
    s3 = []
    for cat in cats:
        if(y_predict[i] == cat):
            s3.append((1-y_pred3.iloc[i,cat])**2)
        else:
            s3.append((0-y_pred3.iloc[i,cat])**2)
    r3.append(sum(s3))
t3 = sum(r3)
rase3 = sqrt(t3/(2*(len(y_test))))
print('The RASE value for Logistic Model is', rase3)

#The RMS metric for Logistic Regression
rms = sqrt(mean_squared_error(y_test, y_pred3_prob))
print('The RMS value for Logistic model is', rms)

#Checking with threshold for events
f = []
for i in (y_pred3_prob):
    if i >= train.iloc[1]:
        f.append(1)
    else:
        f.append(0)

#The misclassification rate for Logistic Regression
error_rate = zero_one_loss(y_test, f)
print('The misclassification rate of the Logistic model is', error_rate)

# #Q2.d) Plot ROC curves

plt.plot(fpr1, tpr1, marker = 'o', color = 'orange', linestyle = 'solid', linewidth = 1, markersize = 6, label = 'k Nearest Neighbor')
plt.plot(fpr2, tpr2, marker = 'o', color = 'blue', linestyle = 'solid', linewidth = 1, markersize = 6, label = 'Decision Tree')
plt.plot(fpr3, tpr3, marker = 'o', color = 'purple', linestyle = 'solid', linewidth = 1, markersize = 3, label = 'Logistic Regression')
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':', label = 'Ref. Line')
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.legend(loc = 'upper left')
plt.title('Receiver Operating Characteristic curve')
plt.grid(True)
plt.show()

#2.e) Accumulative lift charts

def compute_lift_coordinates (
        DepVar,          # The column that holds the dependent variable's values
        EventValue,      # Value of the dependent variable that indicates an event
        EventPredProb,   # The column that holds the predicted event probability
        Debug = 'N'):    # Show debugging information (Y/N)

    # Find out the number of observations
    nObs = len(DepVar)

    # Get the quantiles
    quantileCutOff = np.percentile(EventPredProb, np.arange(0, 100, 10))
    nQuantile = len(quantileCutOff)

    quantileIndex = np.zeros(nObs)
    for i in range(nObs):
        iQ = nQuantile
        EPP = EventPredProb.iloc[i]
        for j in range(1, nQuantile):
            if (EPP > quantileCutOff[-j]):
                iQ -= 1
        quantileIndex[i] = iQ

    # Construct the Lift chart table
    countTable = pd.crosstab(quantileIndex, DepVar)
    decileN = countTable.sum(1)
    decilePct = 100 * (decileN / nObs)
    gainN = countTable[EventValue]
    totalNResponse = gainN.sum(0)
    gainPct = 100 * (gainN /totalNResponse)
    responsePct = 100 * (gainN / decileN)
    overallResponsePct = 100 * (totalNResponse / nObs)
    lift = responsePct / overallResponsePct

    LiftCoordinates = pd.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                    axis = 1, ignore_index = True)
    LiftCoordinates = LiftCoordinates.rename({0:'Decile N',
                                              1:'Decile %',
                                              2:'Gain N',
                                              3:'Gain %',
                                              4:'Response %',
                                              5:'Lift'}, axis = 'columns')

    # Construct the Accumulative Lift chart table
    accCountTable = countTable.cumsum(axis = 0)
    decileN = accCountTable.sum(1)
    decilePct = 100 * (decileN / nObs)
    gainN = accCountTable[EventValue]
    gainPct = 100 * (gainN / totalNResponse)
    responsePct = 100 * (gainN / decileN)
    lift = responsePct / overallResponsePct

    accLiftCoordinates = pd.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                       axis = 1, ignore_index = True)
    accLiftCoordinates = accLiftCoordinates.rename({0:'Acc. Decile N',
                                                    1:'Acc. Decile %',
                                                    2:'Acc. Gain N',
                                                    3:'Acc. Gain %',
                                                    4:'Acc. Response %',
                                                    5:'Acc. Lift'}, axis = 'columns')
        
    if (Debug == 'Y'):
        print('Number of Quantiles = ', nQuantile)
        print(quantileCutOff)
        _u_, _c_ = np.unique(quantileIndex, return_counts = True)
        print('Quantile Index: \n', _u_)
        print('N Observations per Quantile Index: \n', _c_)
        print('Count Table: \n', countTable)
        print('Accumulated Count Table: \n', accCountTable)

    return(LiftCoordinates, accLiftCoordinates)

lift_coordinates, acc_lift_coordinates = compute_lift_coordinates (
        DepVar = score_test_knn['CLAIM_FLAG'],
        EventValue = 1,
        EventPredProb = score_test_knn[1],
        Debug = 'Y')

lift_knn = lift_coordinates
acc_knn = acc_lift_coordinates

lift_coordinates, acc_lift_coordinates = compute_lift_coordinates (
        DepVar = score_test_lr['CLAIM_FLAG'],
        EventValue = 1,
        EventPredProb = score_test_lr[1],
        Debug = 'Y')

lift_lr = lift_coordinates
acc_lr = acc_lift_coordinates

lift_coordinates, acc_lift_coordinates = compute_lift_coordinates (
        DepVar = score_test_dt['CLAIM_FLAG'],
        EventValue = 1,
        EventPredProb = score_test_dt[1],
        Debug = 'Y')

lift_dt = lift_coordinates
acc_dt = acc_lift_coordinates

# Draw the Accumulative Lift chart
plt.plot(acc_knn.index, acc_knn['Acc. Lift'], marker = 'o',
         color = 'orange', linestyle = 'solid', linewidth = 2, markersize = 6, label = 'k Nearest Neighbors')
plt.plot(acc_dt.index, acc_dt['Acc. Lift'], marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6, label = 'Decision Tree')
plt.plot(acc_lr.index, acc_lr['Acc. Lift'], marker = 'o',
         color = 'purple', linestyle = 'solid', linewidth = 2, markersize = 6, label = 'Logistic Regression')

plt.title("Testing Partition Acc. Lift")
plt.grid(True)
plt.xticks(np.arange(1,11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Accumulated Lift")
plt.legend(loc = 'upper right')
plt.show()