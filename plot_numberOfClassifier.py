import pandas as pd
import matplotlib.pyplot as plt
import os

'''
Plotting the experiments number of classifiers.
'''

# Fix constants.
DIRECTMEASURES = ['test accuracy', 'training accuracy', 'micro f1', 'macro f1', 'oobscore', 'strength', 'correlation']
COMPAREMEASURES = ['mcnemar', 'SE', 'upper', 'downer']
LABEL_MODIFY = '_modify'
LABEL_SKLEARN = '_sklearn'
LABEL_REGRESSION = '_regression'
MODIFICATIONS = ['ttfffff', 'tftffff', 'tfftfff', 'tffftff', 'tfffftf', 'tffffft']
VARIANTS = ['log', 'all', 'sqrt']
DATASETS = ['avila', 'car', 'glass', 'heart']


# Here we get the exact path of this script to be able to load the data from it.
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []

# Set the font of the plots to 25.
plt.rcParams.update({'font.size': 25})

# Iterate through every combination.
for name in DATASETS:
    for mtry in VARIANTS:
        for s_str in MODIFICATIONS:
            # We try except the next line, if one file is not existent we do not stop the loops.
            try:
                # Load the csv in as a pd.Dataframe.
                measur = pd.read_csv(f'{user_paths[0]}/data_numberOfClassifiers/classifiers_{name}_{mtry}_{s_str}.csv')

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

                # Up to which number is the plot displayed.
                stop_numbers = list(measur['numberOfClassifier'])[:100]


                # To create a title for the plot.
                PARAMETER = {'max_features': {mtry}, 'splitter': 'best', 'weight_features':WEIGHT_FEATURE}
                BASECLASSIFIER = 'ownDT'
                str1 = f'Baseclassifier:{BASECLASSIFIER} and parameter {PARAMETER} \n + oobcompute={OOBCOMPUTE},cart={CART},c4_5={C4_5},chaid={CHAID},new_features={NEW_FEATURES},Feature|multiply={FEATURES_MULTIPLY},rotate={FEATURES_ROTATE},linearComb={FEATURES_LINEARCOMBINATION} , weight_ensemble={WEIGHT_ENSEMBLE},weight_features={WEIGHT_FEATURE}'
                str2 = f'dataset: {name} & number of trees: Steps: {stop_numbers[0]} total:{stop_numbers[-1]}'
                #str1 = ''#f' Experiment over the Number Of Classifiers with dataset {name}.'
                #str2 = f' Experiment over the Number Of Classifiers with dataset {name}.'


                # Plot every measure to the number we decided before in line 50.
                fig, ax = plt.subplots(nrows=9, ncols=1, figsize=(15, 30), sharex='row')
                #plt.suptitle(str1 + '\n' + str2)
                n = len(DIRECTMEASURES)
                for i in range(n):
                    #if i >= 5: continue
                    ax[i].plot(stop_numbers, measur[DIRECTMEASURES[i]][:len(stop_numbers)],linewidth=4)
                    ax[i].plot(stop_numbers, measur[DIRECTMEASURES[i] + LABEL_MODIFY][:len(stop_numbers)],linewidth=4)
                    if DIRECTMEASURES[i] in ['test accuracy', 'training accuracy', 'micro f1', 'macro f1']:  # , 'oobscore']:
                        ax[i].plot(stop_numbers, measur[DIRECTMEASURES[i] + LABEL_SKLEARN][:len(stop_numbers)],linewidth=4)
                        ax[i].plot(stop_numbers, measur[DIRECTMEASURES[i] + LABEL_REGRESSION][:len(stop_numbers)],linewidth=4)
                        ax[i].set_ylabel(DIRECTMEASURES[i])
                        ax[i].legend(['rf_normal','rf_modify', 'rf_sklearn', 'regression'],loc='lower right') #
                    else:
                        ax[i].set_ylabel(DIRECTMEASURES[i])
                        if i == 4:
                            ax[i].legend(['rf_normal', 'rf_modify'], loc='lower right')
                        else:
                            ax[i].legend(['rf_normal', 'rf_modify'],loc='upper left')

                START = 0
                ax[n].plot(stop_numbers[START:], measur['mcnemar'][:len(stop_numbers)],linewidth=4)
                ax[n].plot(stop_numbers[START:], [3.84 for i in stop_numbers[START:]][:len(stop_numbers)],linewidth=4)
                ax[n].legend(['mcnemar', 'limit'])
                ax[n].set_ylabel('mcnemar')
                ax[n + 1].plot(stop_numbers[START:], measur['e-e_modify'][:len(stop_numbers)],linewidth=4)
                ax[n + 1].plot(stop_numbers[START:], measur['upper'][:len(stop_numbers)],linewidth=4)
                ax[n + 1].plot(stop_numbers[START:], measur['downer'][:len(stop_numbers)],linewidth=4)
                ax[n + 1].set_ylabel('confidence interval')
                ax[n + 1].set_ylabel('confidence interval')
                ax[n + 1].legend(['e-e_modify', 'upper limit', 'down limit'],loc='upper right')
                ax[-1].set_xlabel('number of estimators')
                fig.tight_layout()

                # Decided if we save or show our plot.
                plt.savefig(f'{user_paths[0]}/figures_numberOfClassifiers/plot_classifiers_inter_err_{name}_{mtry}_{s_str}.png', dpi=100)
                #plt.show()
                plt.close(fig)
            except:
                print(f'error: {s_str} of dataset {name}')
