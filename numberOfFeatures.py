import pandas as pd
import measures
from randomforest import randomForestClassifier_own
from chooseData import generator
from decisiontree import Decisiontree
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import time
import math
import datetime
import os
import sys



'''
Experiment over the number of features.
'''


# Fix constants.
DIRECTMEASURES = ['test accuracy', 'training accuracy', 'micro f1', 'macro f1', 'oobscore', 'strength', 'correlation']
COMPAREMEASURES = ['mcnemar',  'e-e_modify', 'upper', 'downer']
LABEL_MODIFY = '_modify'
LABEL_SKLEARN = '_sklearn'
LABEL_REGRESSION = '_regression'
BASECLASSIFIER = Decisiontree
WALLTIME = 172800
PERCENT = 0.8
NUMBER = 100
MAX_NUMBER_FEATURES = 41
REPEATS = 50
MTRY = 'log'


# Catch the input we give the experiment per shell command else we use the parameter in the else row.
featuresToDo = sys.argv
del featuresToDo[0]

if len(featuresToDo) >0:
    s_str = featuresToDo[0]
    del featuresToDo[0]
else:
    s_str = 'tffffff'
    # OOB|HYBRID|MULTI|ROTATE|LK|WEIGHTFEAT|WEIGHTENSEMBLE


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

# Parameter for the basic classifier to use.
PARAMETER = {'max_features': MTRY , 'splitter': 'best', 'weight_features': WEIGHT_FEATURE}

# Initialize the dictonary for all measures.
measur = {}
for item in DIRECTMEASURES:
    measur[item] = []
    measur[item + LABEL_MODIFY] = []
    if item in ['test accuracy', 'training accuracy', 'micro f1', 'macro f1']:
        measur[item + LABEL_SKLEARN] = []
        measur[item + LABEL_REGRESSION] = []
for item in COMPAREMEASURES:
    measur[item] = []


# name of the data set we want to use.
name = 'generator'

# Output in the command line for overview.
print('________________________________________________________')
str1 = f'Baseclassifier:{BASECLASSIFIER} and parameter {PARAMETER} \n + oobcompute={OOBCOMPUTE},cart={CART},c4_5={C4_5},chaid={CHAID},new_features={NEW_FEATURES},Feature|multiply={FEATURES_MULTIPLY},rotate={FEATURES_ROTATE},linearComb={FEATURES_LINEARCOMBINATION} , weight_ensemble={WEIGHT_ENSEMBLE},weight_features={WEIGHT_FEATURE}'
print(str1)
str2 = f'datasets: {name} & number of trees: {NUMBER}'
print(str2)
print(f'normal vs  oobcompute={OOBCOMPUTE},cart={CART},c4_5={C4_5},chaid={CHAID},new_features={NEW_FEATURES},Feature|multiply={FEATURES_MULTIPLY},rotate={FEATURES_ROTATE},linearComb={FEATURES_LINEARCOMBINATION}, weight_ensemble={WEIGHT_ENSEMBLE},weight_features={WEIGHT_FEATURE}')


# Get the function handle from the chooseData class and use it to get the training and test set.
[all_train, y_train, all_test, y_test,cat_ix_var] = generator(max_number_features=MAX_NUMBER_FEATURES)


# The numbers wie use for the experiment and iterate through it.
NUMBERS= range(20,MAX_NUMBER_FEATURES,5)
index = []
maintime = time.time()
for numberOfFeatures in NUMBERS:
    print(f'run: {numberOfFeatures}')

    # Create the two sets with the number of features we want.
    train = all_train[[str(i) for i in range(numberOfFeatures)]]
    test = all_test[[str(i) for i in range(numberOfFeatures)]]

    # Loop for the different attempts.
    for i in range(REPEATS):

        # Initialize the four models and train all models with the training data.
        rf = randomForestClassifier_own(BASECLASSIFIER, PARAMETER, oobcompute=OOBCOMPUTE, cart=True, c4_5=False, chaid=False, featuresMultiply=False, featuresRotate=False, featuresLc=False, weighted=False)
        rf_modify = randomForestClassifier_own(BASECLASSIFIER, PARAMETER, oobcompute=OOBCOMPUTE, cart=CART, c4_5=C4_5, chaid=CHAID, featuresMultiply=FEATURES_MULTIPLY, featuresRotate=FEATURES_ROTATE, featuresLc=FEATURES_LINEARCOMBINATION, weighted=WEIGHT_ENSEMBLE)
        rf.fit(train, y_train, perCent=PERCENT, number=NUMBER, cat_ix=cat_ix_var)
        rf_modify.fit(train, y_train, perCent=PERCENT, number=NUMBER, cat_ix=cat_ix_var)
        rf_sklearn = RandomForestClassifier(criterion='gini', oob_score=True, n_estimators=NUMBER)
        rf_sklearn.fit(train, y_train)
        reg = linear_model.LinearRegression()
        reg.fit(train, y_train)

        # In the next steps we will evaluate the individual models with the different measures.
        y_pred = rf.predict(test)
        tmp = measures.onceMeasures(y_test, y_pred)
        measur['test accuracy'].append(tmp[0])
        measur['training accuracy'].append(accuracy_score(rf.predict(train), y_train))
        measur['micro f1'].append(tmp[1])
        measur['macro f1'].append(tmp[2])
        measur['oobscore'].append(rf.oobScore)
        measur['strength'].append(rf.strength)
        measur['correlation'].append(rf.correlation)

        y_pred_modify = rf_modify.predict(test)
        tmp = measures.onceMeasures(y_test, y_pred_modify)
        measur['test accuracy' + LABEL_MODIFY].append(tmp[0])
        measur['training accuracy' + LABEL_MODIFY].append(accuracy_score(rf_modify.predict(train), y_train))
        measur['micro f1' + LABEL_MODIFY].append(tmp[1])
        measur['macro f1' + LABEL_MODIFY].append(tmp[2])
        measur['oobscore' + LABEL_MODIFY].append(rf_modify.oobScore)
        measur['strength' + LABEL_MODIFY].append(rf_modify.strength)
        measur['correlation' + LABEL_MODIFY].append(rf_modify.correlation)

        y_pred_sklearn = rf_sklearn.predict(test)
        tmp = measures.onceMeasures(y_test, y_pred_sklearn)
        measur['test accuracy' + LABEL_SKLEARN].append(tmp[0])
        measur['training accuracy' + LABEL_SKLEARN].append(accuracy_score(rf_sklearn.predict(train), y_train))
        measur['micro f1' + LABEL_SKLEARN].append(tmp[1])
        measur['macro f1' + LABEL_SKLEARN].append(tmp[2])
        #measur['oobscore' + LABEL_SKLEARN].append(1-rf_sklearn.oob_score)

        y_pred_reg = reg.predict(test)
        y_pred_reg = y_pred_reg.round(decimals=0, out=None)
        tmp = measures.onceMeasures(y_test, y_pred_reg)
        measur['test accuracy' + LABEL_REGRESSION].append(tmp[0])
        measur['training accuracy' + LABEL_REGRESSION].append(accuracy_score(reg.predict(train).round(decimals=0, out=None), y_train))
        measur['micro f1' + LABEL_REGRESSION].append(tmp[1])
        measur['macro f1' + LABEL_REGRESSION].append(tmp[2])

        tmp = measures.compareMeasures(y_test.to_numpy(), y_pred, y_pred_modify)  # davor)
        if tmp[0] == math.inf:
            tmp[0] = -1
        measur['mcnemar'].append(tmp[0])
        measur['e-e_modify'].append((1- measur['test accuracy'][-1])-(1-measur['test accuracy' + LABEL_MODIFY][-1]))
        upper = tmp[1][1] - tmp[1][2] + 1.96 * (tmp[1][0] + (1 / (2 * NUMBER)))
        downer = tmp[1][1] - tmp[1][2] - 1.96 * (tmp[1][0] + (1 / (2 * NUMBER)))
        measur['upper'].append(upper)
        measur['downer'].append(downer)

        index.append(numberOfFeatures)




print(f'used time: {time.time() - maintime}')

# Save the measures value to a csv file.
measur['numberOfFeatures'] = index #list(NUMBERS)
save_df = pd.DataFrame.from_dict(measur)
save_df = save_df.set_index('numberOfFeatures')
save_df.to_csv(f'{user_paths[0]}/data_numberOfFeatures/features_{name}_{MTRY}_{s_str}.csv')



