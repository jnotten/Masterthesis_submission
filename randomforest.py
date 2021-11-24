from distython import HEOM
import math
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from queue import Queue
from sklearn.metrics import accuracy_score
import operator

'''
The class for our modifiable random forest. 
We have two public object methods: fit and predict with them we can train our models and predict the datas.
In our constructor we pass all the parameters to the individual modifications, so we can set the random forest exactly
as we want it. In addition, all object auxiliary variables are initialised and some are already declared.
'''

class randomForestClassifier_own:
    def __init__(self, Baseclassifier, parameter, oobcompute=True, number_newFeatures=None, featuresMultiply=False,
                 featuresRotate=False, featuresLc=False, cart=True, c4_5=False, chaid=False, weighted=False):
        '''
        This is the constuctor of the class and sets all the modifications that we receive.
        :param Baseclassifier: The basic classifier we use in our ensemble.
        :param parameter: The parameter we use for our basic classifier.
        :param oobcompute: A boolean if we want to use the oob measures.
        :param number_newFeatures: If you want a alternative number of new features in the features engineering.
        :param featuresMultiply: A boolean if we want to use the multiply of the features engineering.
        :param featuresRotate: A boolean if we want to use the rotate of the features engineering.
        :param featuresLc: A boolean if we want to use the linear combination of the features engineering.
        :param cart: A boolean if we want to use the CART algorithm in the decision tree level.
        :param c4_5: A boolean if we want to use the C4.5 algorithm in the decision tree level.
        :param chaid: A boolean if we want to use the CHAID algorithm in the decision tree level.
        :param weighted: A boolean if we want to use the dynamical integration for our model.
        '''
        self.ensemble = []
        self.Baseclassifier = Baseclassifier
        self.parameter = parameter
        self.oobcompute = oobcompute
        self.OOB_Set = []
        self.OOB_sol = []
        self.weighted = weighted
        self.weight = []
        self.data = None
        self.oobScore = 0
        self.strength = 0
        self.correlation = 0
        self.cart = cart
        self.c4_5 = c4_5
        self.chaid = chaid
        self.numberNewFeatures = number_newFeatures
        self.features_multiply = featuresMultiply
        self.features_rotate = featuresRotate
        self.features_linearcombination = featuresLc
        self.new_features = self.features_multiply or self.features_rotate or self.features_linearcombination
        self.weight_features = None
        self.combinations = None
        self.combinationsRotate = None
        self.degree = None
        self.linearcombintion = None
        self.linearcombintionCoefficients = None
        self.train = None
        self.cat_ix = None
        self.weight_features = None

    def fit(self, X, y, perCent=0.8, number=50, cat_ix=None):
        '''
            The main fit method to train our model with the given parameter. Per default we get a ensemble with size of
             50 classifierts and use 80 per cent of the given training set for each basic classifier.

        :param X: The training set.
        :param y: The true values of the instances in the training set.
        :param perCent: The per cent how much of the training set we use in the bagging method.
        :param number: Number of classifiers for the ensemble.
        :param cat_ix: A list of the features that are categorical.

        '''


        # we start with the feature engineering. First we check if we want to use it and calculate the maximum number of
        # new features under the variable max_features and calculate the combinations to be tested, i.e. all.
        # In rotation, drawn the certain number from all possibilities out the combinations.
        if self.new_features:
            if self.train is None:
                if self.numberNewFeatures is None:
                    max_features = int(math.log(len(X.columns), 2))
                else:
                    max_features = self.numberNewFeatures

                if self.features_multiply or self.features_rotate or self.features_linearcombination:
                    if self.combinations is None:
                        mesh = np.array(np.meshgrid(X.columns, X.columns))
                        combinations = mesh.T.reshape(-1, 2)

                if self.features_rotate:
                    if self.combinationsRotate is None:
                        combinationsRotate = combinations
                        indiz = np.random.choice([i for i in range(len(combinationsRotate))],size=max_features)
                        self.combinationsRotate = combinationsRotate[indiz]
                        if self.degree is None:
                            self.degree = [np.random.uniform(-90, 90) for i in range(len(combinationsRotate))]

                if self.features_linearcombination:
                    if self.linearcombintion is None:
                        mesh = np.array(np.meshgrid(X.columns, X.columns,X.columns))
                        linearcombinations = mesh.T.reshape(-1, 3)


                # We copy our training set so that we do not change the original data set.
                self.train = X.copy()


                # Create our test data set to test the different combinations and calculate their chi-square value.
                # Keep only the highest max_feature combinations in terms of chi-square value.
                # Then save all the combinations and coefficients that remain in the object variables.
                test_size = 300
                if len(self.train) <= test_size:
                    testdf = self.train.copy()
                else:
                    testdf = self.train.sample(n=test_size)
                if self.features_multiply:
                    chi_value = {}
                    for arr in combinations:
                        testdf[str(arr[0])+'*'+str(arr[1])] = testdf[arr[0]] * testdf[arr[1]]
                        if arr[0] not in cat_ix or arr[1] not in cat_ix:
                            chi_value[str(arr[0])+'*'+str(arr[1])] = self.__chi_square_test(pd.cut(testdf[(str(arr[0]) + '*' + str(arr[1]))], 7), y[testdf.index])
                        else:
                            chi_value[str(arr[0])+'*'+str(arr[1])] = self.__chi_square_test(testdf[(str(arr[0]) + '*' + str(arr[1]))], y[testdf.index])
                        if len(chi_value) > max_features:
                            delkey = min(chi_value, key=chi_value.get)
                            del chi_value[delkey]
                            del testdf[delkey]
                    self.combinations = []
                    for item in chi_value.keys():
                        self.combinations.append(item.split('*'))
                        self.train[item] = self.train[item.split('*')[0]] * self.train[item.split('*')[1]]
                if self.features_linearcombination:
                    chi_value = {}
                    for arr in linearcombinations:
                        # Create uniform at random the coefficient of the linear combination.
                        coeff = [np.random.uniform(-1,1) for i in range(3)]
                        testdf[str(coeff[0]) +'*'+str(arr[0])+'+'+str(coeff[1]) +'*'+str(arr[1])+'+'+str(coeff[2]) +'*'+str(arr[2])]  = coeff[0]*testdf[arr[0]] + coeff[1]*testdf[arr[1]] + coeff[2]*testdf[arr[2]]
                        if arr[0] not in cat_ix or arr[1] not in cat_ix or arr[2] not in cat_ix:
                            chi_value[str(coeff[0]) +'*'+str(arr[0])+'+'+str(coeff[1]) +'*'+str(arr[1])+'+'+str(coeff[2]) +'*'+str(arr[2])] = self.__chi_square_test(pd.cut(testdf[str(coeff[0]) + '*' + str(arr[0]) + '+' + str(coeff[1]) + '*' + str(arr[1]) + '+' + str(coeff[2]) + '*' + str(arr[2])], 7), y[testdf.index])
                        else:
                            chi_value[str(coeff[0]) +'*'+str(arr[0])+'+'+str(coeff[1]) +'*'+str(arr[1])+'+'+str(coeff[2]) +'*'+str(arr[2])] = self.__chi_square_test(testdf[str(coeff[0]) + '*' + str(arr[0]) + '+' + str(coeff[1]) + '*' + str(arr[1]) + '+' + str(coeff[2]) + '*' + str(arr[2])], y[testdf.index])
                        if len(chi_value) > max_features:
                            delkey = min(chi_value, key=chi_value.get)
                            del chi_value[delkey]
                            del testdf[delkey]
                    self.linearcombintion = []
                    self.linearcombintionCoefficients = []
                    for item in chi_value.keys():
                        tmparr =  item.split('+')
                        coeff = []
                        feat = []
                        for item2 in tmparr:
                            coeff.append(item2.split('*')[0])
                            feat.append(item2.split('*')[1])

                        self.linearcombintion.append(feat)
                        self.linearcombintionCoefficients.append(coeff)
                        self.train[item] = float(coeff[0]) * self.train[feat[0]] + float(coeff[1]) *self.train[feat[1]] + float(coeff[2]) *self.train[feat[2]]
                if self.features_rotate:
                    i = 0
                    for arr in self.combinationsRotate:
                        tmp = self.degree[i]
                        self.train['rotate:' + str(arr[0]) + 'on plane' + str(arr[0]) + str(arr[1]) + 'angle:' + str(tmp)] = math.cos(tmp)*self.train[arr[0]]
                        self.train['rotate:' + str(arr[1]) + 'on plane' + str(arr[0]) + str(arr[1]) + 'angle:' + str(tmp)] = math.sin(tmp)*self.train[arr[1]]
                        i = i+1

            X = self.train
        else:
            # Otherwise, give the reference as an object variable.
            if self.train is None:
                self.train = X

        # Change the data sets to numpy arrays and change the categorical features to the matching indices.
        if cat_ix is not None:
            self.cat_ix = []
            for item in cat_ix:
                self.cat_ix.append(list(X.columns).index(item))

        que = Queue()
        X = X.to_numpy()
        y = y.to_numpy()


        # Creat a list with all split algorithms we want to use.
        splitMethods = []
        if self.cart:
            splitMethods.append('CART')
        if self.c4_5:
            splitMethods.append('C4.5')
        if self.chaid:
            splitMethods.append('CHAID')

        # Here we create the individual classifiers with the respective splitmethods,
        # which iterate through their list as appropriate.
        j = 0
        for i in range(number):
            self.parameter['splitMethod'] = splitMethods[j]
            self.__fitOneTree(que, X, y, (len(X) * perCent), cat_ix)
            j = j+1
            if j >= len(splitMethods):
                j=0

        # We get the classifiers back in a queue and add them to our list.
        # This is due to the preparation for the step towards multithreading.
        while not que.empty():
            self.ensemble.append(que.get())


        # The call of the computeOOB method if we want it.
        if self.oobcompute:
            self.__computeOOB(X, y)

    def __chi_square_test(self, x, y):
        '''
        With the help of a pandas method we calculate the chi-square value of two lists.
        :param x: list x.
        :param y: list y.
        :return: chi-square value of the list x and y.
        '''
        # Create with pandas the frequency matirx of x and y with the totals in the last row and col.
        conc_table = pd.crosstab(x, y,rownames=['x'],colnames=['y'], margins=True)

        # Computed the values to get the value.
        total = conc_table['All']['All']
        cols = conc_table.columns
        rows = conc_table.index
        chiSquare = 0
        for col in cols:
            for row in rows:
                if col == 'All': break
                if row == 'All': continue
                t_ = (conc_table['All'][row] * conc_table[col]['All']) / total
                tmp = (conc_table[col][row] - t_) ** 2
                tmp = tmp / t_
                chiSquare += tmp
        return chiSquare

    def __fitOneTree(self, que, X, y, samplesize, cat_ix):
        '''
        In this method we train one classifier and give them back to out main algorithm.

        :param que: Is the return queue we use.
        :param X: Training set.
        :param y: True values of classes of training set.
        :param samplesize: The size we want to use for each bagging set.
        :param cat_ix: The list of indices of the categorical features.
        :return:
        We return the classifier in the queue que.
        '''


        # We apply the bagging method to the indices and form the tmp_X and tmp_Y the bagging training set.
        indicies = [i for i in range(len(X))]
        tmp_indicies = np.random.choice(indicies, size=int(np.floor(samplesize)), replace=True)
        tmp_X = [list(X[item]) for item in tmp_indicies]
        tmp_y = [y[item] for item in tmp_indicies]


        # The proberties for the feature weighting are calculated or an equal
        # distribution is calculated, depending on the desired modification.
        self.weight_features_bool = self.parameter['weight_features']
        if self.weight_features_bool:
            pd_tmp_X = pd.DataFrame(tmp_X)
            pd_tmp_y = pd.Series(tmp_y)
            total_weight = 0
            feature_weight = []
            for key in pd_tmp_X:
                # If a feature is numeric, it is made categorical using a pandas method.
                if key not in cat_ix:
                    pd_tmp_X[key] = pd.cut(pd_tmp_X[key], int(len(tmp_X)/10))
                corr_sum = self.__chi_square_test(pd_tmp_X[key], pd_tmp_y)
                feature_weight.append(math.sqrt(corr_sum))
                total_weight += math.sqrt(corr_sum)
            test_sum = 0
            for i in range(len(feature_weight)):
                feature_weight[i] = feature_weight[i] / total_weight
                test_sum += feature_weight[i]
            del pd_tmp_X
            self.weight_features = feature_weight
        else:
            n = len(self.train.columns)
            self.weight_features = [1 / n for i in range(n)]

        # Here we calculate the OOB set by throwing out all the instances in the IOB set from the complete set.
        for item in tmp_indicies:
            if item in indicies:
                indicies.pop(indicies.index(item))
        self.OOB_Set.append(np.array(indicies))

        # If we don't use our own framework for the basic algorithm, we get rid of two parameters.
        # Then fit the classifier with the parameters and bagging set.
        if self.Baseclassifier == DecisionTreeClassifier:
            del self.parameter['splitMethod']
            del self.parameter['weight_features']
        tmp_classifier = self.Baseclassifier(**self.parameter)
        tmp_classifier.fit(tmp_X, tmp_y, self.weight_features)

        # Precalculate some values for the dynamical integration.
        if self.weighted:
            weights_tmp = pd.DataFrame([0 for i in range(len(X))])
            weights_tmp[0][self.OOB_Set[-1]] = 1
            y_oob = tmp_classifier.predict(X[self.OOB_Set[-1]])
            y_comparison = y[self.OOB_Set[-1]]
            weights_tmp[0][self.OOB_Set[-1][abs(y_oob - y_comparison) == 1]] = -1
            self.weight.append(weights_tmp[0].to_numpy())

        # Put the finished classifier on the return queue.
        que.put(tmp_classifier)

    def __computeOOB(self, X, y):
        '''
        Object method to compute the OOB score, strength and correlation measures.
        :param X: Training set.
        :param y: True values of classes of training set.
        '''

        number = len(self.ensemble)
        prob = [0 for i in range(number)]
        prob_ = [0 for i in range(number)]
        sum_strength = 0
        sum_corr = 0
        y_pred = []

        # Precalculate data if we only add classifiers to our ensemble.
        data = self.data
        if self.data is None:
            data = np.empty((0,len(X),))
            data = pd.DataFrame(data)

        # For every add classifier we predict his OOB set and save it in the data variable.
        n = len(self.OOB_Set)
        for i in range(number):
            if i >= len(self.OOB_sol)-1:
                data = data.append([np.nan],ignore_index=True)
                pred = self.ensemble[i].predict(X[self.OOB_Set[i]])
                self.OOB_sol.append(pred)
            else:
                pred = self.OOB_sol[i]
                continue
            for j in range(len(self.OOB_Set[i])):
                data[self.OOB_Set[i][j]][i] = pred[j]


        # Now we calculate the Q -values and the different for the strength and correlation measures and we search the
        # most predicted class of the classifiers without the instance x_i in the bagging set.
        for i in range(len(X)):
            tmp = data[i].value_counts()
            sort = tmp.index
            if len(sort) > 0:
                y_pred.append(sort[0])
            else:
                y_pred.append(-1)

            Q = 0
            Qmax = 0
            j_ = None
            for item in tmp.items():
                if Q and Qmax: break
                if item[0] == y[i]:
                    Q = item[1]
                elif Qmax == 0:
                    Qmax = item[1]
                    j_ = item[0]

            sum_strength += Q - Qmax
            sum_corr += (Q - Qmax) ** 2


            should = data[i].notnull()
            for j in range(number):
                if not should[j]:
                    continue
                if data[i][j] == y[i]:
                    prob[j] +=1
                if data[i][j] == j_:
                    prob_[j] += 1


        # At least combine the whole coefficient to the strength, correlation measure and the oobscore.
        n = len(X)

        self.strength = sum_strength / n
        sum_corr = sum_corr / n
        corr = sum_corr - (self.strength**2)
        corr2 = 0
        for i in range(number):
            div = len(self.OOB_Set[i])
            prob[i] = prob[i]/div
            prob_[i] = prob_[i] / div
            corr2 += np.sqrt(prob[i]+prob_[i]+((prob[i]-prob_[i])**2))
        corr2 = corr2/number
        corr2 = corr2 ** 2
        self.correlation = corr/corr2
        self.oobScore = accuracy_score(y, y_pred)
        self.data = data

    def predict(self,X):
        '''
        With this public object method we get a data set X and predicted them with our model.
        :param X: Data set we want to predict.
        :return: A list of predicted classes in it.
        '''

        # First we must calculate the new features of the features engineering so we can split on them.
        if self.new_features:
            test = X.copy()

            if self.features_multiply:
                for arr in self.combinations:
                    test[str(arr[0])+'*'+str(arr[1])] = X[arr[0]] * X[arr[1]]

            if self.features_rotate:
                i = 0
                for arr in self.combinationsRotate:
                    tmp = self.degree[i]
                    test['rotate:' + str(arr[0]) + 'on plane' + str(arr[0]) + str(arr[1]) + 'angle:' + str(tmp)] = math.cos(tmp)*test[arr[0]]
                    test['rotate:' + str(arr[1]) + 'on plane' + str(arr[0]) + str(arr[1]) + 'angle:' + str(tmp)] = math.sin(tmp)*test[arr[1]]
                    i = i+1

            if self.features_linearcombination:
                for i in range(len(self.linearcombintion)):
                    feat = self.linearcombintion[i]
                    coeff = self.linearcombintionCoefficients[i]
                    test[str(coeff[0]) +'*'+str(feat[0])+'+'+str(coeff[1]) +'*'+str(feat[1])+'+'+str(coeff[2]) +'*'+str(feat[2])] = float(coeff[0]) * test[feat[0]] + float(coeff[1]) *  test[feat[1]] + float(coeff[2]) *  test[feat[2]]

            X = test


        # Then we start to predict with our basic classifiers.
        # If we weight the ensemble with dynamical integration we train out HEOM metric first.
        # Then we get a big distance matrix from all dataset we want to predict to our training set
        # (OOB set is subset of training set).
        solution = []
        weights = []
        X = np.array(X)

        if self.weighted:
            tmp_train = self.train.to_numpy()

            if not len(self.cat_ix) == 0:

                all = np.concatenate((X, tmp_train))
                heom_metric = HEOM((all), cat_ix=self.cat_ix)
                M = []
                for item1 in X:
                    M.append([])
                    for item2 in tmp_train:
                        M[-1].append(1/heom_metric.heom(np.array(item1), np.array(item2)))
            else:
                from scipy.spatial import distance_matrix
                M = distance_matrix(X, tmp_train)


        # Then we get all predicted classes of the classifiers and the weight we get.
        i = 0
        for tree in self.ensemble:
            if self.weighted:
                term = self.weight[i] * M
                term1 = term.sum(axis=1)
                term2 = (abs(term)).sum(axis=1)
                ergebnis = term1/term2
                weights.append(ergebnis)

            solution.append(tree.predict(X))

            i +=1


        # Now we search the biggest value of out solutions with regard to the weights.
        tmp_y = pd.DataFrame(solution)
        y_pred = []
        if self.weighted:
            tmp_weight = pd.DataFrame(weights)
            for i in tmp_y.columns:
                pred_class = {}
                for j in range(len(self.ensemble)):
                    if tmp_y[i][j] in pred_class.keys():
                        pred_class[tmp_y[i][j]] += tmp_weight[i][j]
                    else:
                        pred_class[tmp_y[i][j]] = tmp_weight[i][j]

                y_pred.append(max(pred_class.items(), key=operator.itemgetter(1))[0])

        else:
            # Or the most often predicted class.
            for i in tmp_y.columns:
                y_pred.append(tmp_y[i].mode()[0])

        return y_pred






