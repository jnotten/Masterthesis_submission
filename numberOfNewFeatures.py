import pandas as pd
import measures
from randomforest import randomForestClassifier_own
from chooseData import DATASETS
from decisiontree import Decisiontree
from sklearn.metrics import accuracy_score
import time
import math
import datetime
import os
import sys

'''
Experiment over the number of new features.
'''

# Fix constants.
DIRECTMEASURES = ['test accuracy', 'training accuracy', 'micro f1', 'macro f1', 'oobscore', 'strength', 'correlation']
COMPAREMEASURES = ['mcnemar', 'e-e_modify', 'upper', 'downer']
LABEL_MODIFY = '_modify'
BASECLASSIFIER = Decisiontree
WALLTIME = 172800
PERCENT = 0.8
NUMBER = 10
MAX_NUMBER_FEATURES = 20
REPEATS = 50

# Catch the input we give the experiment per shell command else we use the parameter in the else row.
featuresToDo = sys.argv
del featuresToDo[0]

if len(featuresToDo) >0:
    s_str = featuresToDo[0]
    del featuresToDo[0]
else:
    s_str = 'tffffff'
    # OOB|HYBRID|MULTI|ROTATE|LK|WEIGHTFEAT|WEIGHTENSEMBLE

if len(featuresToDo) == 0:
    featuresToDo = ['heart']

# Here we get the exact path of this script to be able to load the data from it.
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []

#now = datetime.datetime.now().strftime('%Y_%m_%d')#__%H_%M_%S')

# Change the string with modifications to booleans.
OOBCOMPUTE = s_str[0] == 't'
CART = True
C4_5 = s_str[1] == 't'
CHAID = s_str[1] == 't'
NEW_FEATURES = True
FEATURES_MULTIPLY = s_str[2] == 't'
FEATURES_ROTATE = s_str[3] == 't'
FEATURES_LINEARCOMBINATION = s_str[4] == 't'
WEIGHT_FEATURE= s_str[5] == 't'
WEIGHT_ENSEMBLE = s_str[6] == 't'
MTRY =  'log'

# Parameter for the basic classifier to use.
PARAMETER = {'max_features': MTRY, 'splitter': 'best', 'weight_features': WEIGHT_FEATURE}

# Initialize the dictonary for all measures.
measur = {}
for item in DIRECTMEASURES:
    measur[item + LABEL_MODIFY] = []
for item in COMPAREMEASURES:
    measur[item] = []
name = featuresToDo[0]

# Output in the command line for overview.
print('________________________________________________________')
str1 = f'Baseclassifier:{BASECLASSIFIER} and parameter {PARAMETER} \n + oobcompute={OOBCOMPUTE},cart={CART},c4_5={C4_5},chaid={CHAID},new_features={NEW_FEATURES},Feature|multiply={FEATURES_MULTIPLY},rotate={FEATURES_ROTATE},linearComb={FEATURES_LINEARCOMBINATION} , weight_ensemble={WEIGHT_ENSEMBLE},weight_features={WEIGHT_FEATURE}'
print(str1)
str2 = f'datasets: {name} & number of trees: {NUMBER}'
print(str2)
print(f'normal vs  oobcompute={OOBCOMPUTE},cart={CART},c4_5={C4_5},chaid={CHAID},new_features={NEW_FEATURES},Feature|multiply={FEATURES_MULTIPLY},rotate={FEATURES_ROTATE},linearComb={FEATURES_LINEARCOMBINATION}, weight_ensemble={WEIGHT_ENSEMBLE},weight_features={WEIGHT_FEATURE}')

maintime = time.time()

# Get the function handle from the chooseData class and use it to get the training and test set.
func = DATASETS[name]
[train, y_train, test, y_test, cat_ix_var] = func()

# The numbers wie use for the experiment.
NUMBERS= range(1,MAX_NUMBER_FEATURES,1)
index = []
# Loop for the different attempts.
for i in range(REPEATS):

    # Initalize a random forest model without modifications, train and test it. We save the accuracy for the compare measures.
    rf = randomForestClassifier_own(BASECLASSIFIER, PARAMETER, oobcompute=OOBCOMPUTE, cart=True, c4_5=False, chaid=False, featuresMultiply=False, featuresRotate=False, featuresLc=False, weighted=False)
    rf.fit(train, y_train, perCent=PERCENT, number=NUMBER, cat_ix=cat_ix_var)
    y_pred = rf.predict(test)
    tmp = measures.onceMeasures(y_test, y_pred)
    measurTestAccuracyCompareValue = tmp[0]

    tmp_time = time.time()
    # Iterate through the numbers we want to tried in the experiment.
    for numberOfNewFeatures in NUMBERS:

            # Initalize a random forest with the modification, train and predict the test set.
            rf_modify = randomForestClassifier_own(BASECLASSIFIER, PARAMETER, oobcompute=OOBCOMPUTE, cart=CART, c4_5=C4_5, chaid=CHAID, featuresMultiply=FEATURES_MULTIPLY, featuresRotate=FEATURES_ROTATE, featuresLc=FEATURES_LINEARCOMBINATION, weighted=WEIGHT_ENSEMBLE, number_newFeatures=numberOfNewFeatures)
            rf_modify.fit(train, y_train, perCent=PERCENT, number=NUMBER, cat_ix=cat_ix_var)
            y_pred_modify = rf_modify.predict(test)

            # In the next steps we will evaluate the modified model with the different measures and compare the
            # model with the random forest without the modifications.
            tmp = measures.onceMeasures(y_test, y_pred_modify)
            measur['test accuracy' + LABEL_MODIFY].append(tmp[0])
            measur['training accuracy' + LABEL_MODIFY].append(accuracy_score(rf_modify.predict(train), y_train))
            measur['micro f1' + LABEL_MODIFY].append(tmp[1])
            measur['macro f1' + LABEL_MODIFY].append(tmp[2])
            measur['oobscore' + LABEL_MODIFY].append(rf_modify.oobScore)
            measur['strength' + LABEL_MODIFY].append(rf_modify.strength)
            measur['correlation' + LABEL_MODIFY].append(rf_modify.correlation)

            tmp = measures.compareMeasures(y_test.to_numpy(), y_pred, y_pred_modify)
            if tmp[0] == math.inf:
                tmp[0] = -1
            measur['mcnemar'].append(tmp[0])
            measur['e-e_modify'].append((1- measurTestAccuracyCompareValue)-(1-measur['test accuracy' + LABEL_MODIFY][-1]))
            upper = tmp[1][1] - tmp[1][2] + 1.96 * (tmp[1][0] + (1 / (2 * NUMBER)))
            downer = tmp[1][1] - tmp[1][2] - 1.96 * (tmp[1][0] + (1 / (2 * NUMBER)))
            measur['upper'].append(upper)
            measur['downer'].append(downer)

            index.append(numberOfNewFeatures)

print(f'used time: {time.time() - tmp_time}')

# Save the measures value to a csv file.
measur['numberOfFeatures'] = index #list(NUMBERS)
save_df = pd.DataFrame.from_dict(measur)
save_df = save_df.set_index('numberOfFeatures')
save_df.to_csv(f'{user_paths[0]}/data_numberOfNewFeatures/newfeatures_{name}_{now}_{MTRY}_{s_str}.csv')



