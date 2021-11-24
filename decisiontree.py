import math
import numpy as np
from queue import Queue
from scipy.stats import entropy

'''
In this class we define our decision tree models with modifications. The various settings for the modifications are
passed as parameters. You can choose one of the different split methods according to which the tree will be developed.
In addition to the maximum number of features that should be passed, the probability after which the features should 
be drawn can also be passed.
'''

def giniImpurity(x):
    '''
        This method compute the gini impurity value of a list.
    :param x: list x with different values.
    :return: the gini value of the list x.
    '''
    sum = 0
    for item in x:
        sum += item**2
    return (1-sum)

class Decisiontree:
    '''
    Class for the a decicion tree model with different method to fit and predict.
    '''
    def __init__(self,**parameter):
        '''
        Initialise a object of the class decision tree and change the parameter in object variables.
        :param parameter: A dictionary of parameters for our models.
        '''
        self.parameter = parameter
        self.max_features = parameter['max_features']
        self.splitMethod = parameter['splitMethod']
        self.minsample = 0
        self.threshold = []
        self.features = []
        self.target = []
        self.measures = []
        self.nextleft = []
        self.nextright = []

    def fit(self,X,y,weight,**parameter):
        '''
            The class method to fit our model. This is a pipeline to change the sets and
            call the privat buildTree method.
        :param X: Training set.
        :param y: Classes of the training set.
        :param weight: The properties for each feature.
        :param parameter: The rest of the parameter we not use.
        '''
        X = np.array(X)
        y = np.array(y)

        self.weight_features = weight
        self.__buildTree(X, y)

    def predict(self,X):
        '''
            The class method to predict the parameter set X with our data set X.
        :param X: Set we want to predicted.
        :return: List of classes we are predicted.
        '''

        # Create a list of predicted classes with value -1 and a queue with a node index,
        # the whole set X , y_pred and the index as element. These represent our nodes in our tree.
        y_pred = np.array([-1 for i in range(len(X))])
        que = Queue()
        que.put([0,X,y_pred,np.array([i for i in range(len(X))])])


        while not que.empty():
            # Loop as long as we have nodes to compute in the queue

            [actualIndex,X,y_pred_tmp,index] = que.get()

            if self.target[actualIndex] is None:
                # If our node no leaf we check the set we get above if
                # its value are lower equal as the threshold of the node.
                links = X[:, self.features[actualIndex]]<= self.threshold[actualIndex]

                if len(X[links]) > 0:
                    # Put all instances the lower equal of the threshold to the queue with the left child of the node.
                    que.put([self.nextleft[actualIndex],X[links],y_pred_tmp[links],index[links]])

                if len(X[~links]) >0:
                    # Put all instances the greater of the threshold to the queue with the right child of the node.
                    que.put([self.nextright[actualIndex], X[~links],y_pred_tmp[~links],index[~links]])

            else:
                # If node is a leaf every instances of the set X get the predicted class of the node.
                y_pred[index] = self.target[actualIndex]

        return y_pred

    def __buildTree(self, X, y):
        '''
        Private class method to build a tree with the training set X,y.
        :param X: Training set.
        :param y: Classes of the training set.
        '''

        # Create a queue with the representative for nodes and add the root node with the impurity value of the set X.
        que = Queue()
        que.put([X,y,1])
        while not que.empty():
            # Loop as long as queue not empty means no nodes in queue.
            # Take a node representatives from queue.
            [X,y,value]= que.get()

            # If only one instance in set X or if al instances has the same class create the a leaf.
            if len(y) <= self.minsample:
                self.target.append(y[0])
                self.features.append(None)
                self.threshold.append(None)
                self.nextleft.append(None)
                self.nextright.append(None)
            if len(set(y)) == 1:
                self.target.append(y[0])
                self.features.append(None)
                self.threshold.append(None)
                self.nextleft.append(None)
                self.nextright.append(None)
            else:
                # Otherwise we split the set X with the split method of the object.
                feature = threshold = None
                while feature is None:
                    if self.splitMethod == 'CART':
                        [feature,threshold,value1,value2] = self.__splitCART(X, y, value)
                    elif self.splitMethod == 'C4.5':
                        [feature, threshold, value1, value2] = self.__splitC4_5(X, y, value)
                    elif self.splitMethod == 'CHAID':
                        [feature, threshold, value1, value2] = self.__splitCHAID(X, y, value)

                # We get from the splitting method a featues, threshold and the values of the splits.
                # Create a node with the both children.
                self.target.append(None)
                self.features.append(feature)
                self.threshold.append(threshold)
                m = len(self.nextleft)+que.qsize()+1
                self.nextleft.append(m)
                self.nextright.append(m+1)

                # Split the set and put the both children representatives in the queue.
                split1 = X[:,feature]<=threshold
                que.put([X[split1],y[split1],value1])
                split2 = X[:,feature] >threshold
                que.put([X[split2],y[split2],value2])

    def __splitC4_5(self, X, y, par_entropie):
        '''
            Here we check the max_features features to get the best split above them according to the entropie value.
        :param X: Set to split.
        :param y: Classes of the set.
        :param par_entropie: The value of the set.
        :return: Give a feature, threshold and the two entropie values of the split
        '''

        # First we check how much features we should check with the the feature weights.
        numberFeatures = len(X[0])
        if self.max_features == 'sqrt':
            n = math.ceil(math.sqrt(numberFeatures))
            choice = np.random.choice([i for i in range(numberFeatures)], size=n, replace=False, p=self.weight_features)
        elif self.max_features == 'all':
            n = numberFeatures
            choice = [i for i in range(numberFeatures)]
        elif self.max_features == 'log':
            n = math.ceil(math.log2(numberFeatures))
            choice = np.random.choice([i for i in range(numberFeatures)], size=n, replace=False, p=self.weight_features)

        # Now we initialize our return variable.
        classes = list(set(y))
        minentdiff = 0
        minfeature = None
        minthreshold = None
        minent1 = None
        minent2 = None
        for feature in choice:

            # For each feature, we should check, we sort the values in the column of the feature and
            # create a set with all values but without double elements.
            sortind = X[:, feature].argsort()
            tmpX = X[sortind]
            tmpy = y[sortind]
            values = list(set(tmpX[:, feature]))
            for i in range(len(values) - 1):
                # Now we go to all various values of the feature. Create a threshold in the middle of the value and
                # the next value. Then split the set with this threshold and compute the probabilities for both
                # value in the sets. Then compute the entropie value for both sets and if term in 202 is the minimum
                # till now we save it in our return values.
                threshold = (values[i + 1] + values[i]) / 2
                split1 = tmpy[tmpX[:, feature] <= threshold]
                split2 = tmpy[tmpX[:, feature] > threshold]
                if len(split1) == 0 or len(split2) == 0: continue
                probs1 = []
                probs2 = []
                for j in classes:
                    probs1.append(len(split1[split1[:] == j]) / len(split1))
                    probs2.append(len(split2[split2[:] == j]) / len(split2))

                ent1 = entropy(probs1,base=2)
                ent2 = entropy(probs2,base=2)
                entropy_diff = par_entropie - (len(split1) / len(tmpX) * ent1 + len(split2) / len(tmpX) * ent2)
                if entropy_diff > minentdiff:
                    minfeature = feature
                    minthreshold = threshold
                    minentdiff = entropy_diff
                    minent1 = ent1
                    minent2 = ent2

        return [minfeature, minthreshold, minent1, minent2]

    def __splitCHAID(self, X, y):
        '''
            Here we check the max_features features to get the best split above them according to the chi-square value.
        :param X: Set to split.
        :param y: Classes of the set.
        :return: Give a feature, threshold and the two CHAID values of the split
        '''

        # First we check how much features we should check with the the feature weights.
        numberFeatures = len(X[0])
        if self.max_features == 'sqrt':
            n = math.ceil(math.sqrt(numberFeatures))
            choice = np.random.choice([i for i in range(numberFeatures)], size=n, replace=False, p=self.weight_features)
        elif self.max_features == 'all':
            n = numberFeatures
            choice = [i for i in range(numberFeatures)]
        elif self.max_features == 'log':
            n = math.ceil(math.log2(numberFeatures))
            choice = np.random.choice([i for i in range(numberFeatures)], size=n, replace=False, p=self.weight_features)

        # Now we initialize our return variable.
        classes = list(set(y))
        maxChi = 0
        maxfeature = None
        maxthreshold = None
        minchi1 = None
        minchi2 = None
        for feature in choice:

            # For each feature, we should check, we sort the values in the column of the feature and
            # create a set with all values but without double elements
            sortind = X[:, feature].argsort()
            tmpX = X[sortind]
            tmpy = y[sortind]
            values = list(set(tmpX[:, feature]))
            cols_classes = list(set(X[:, feature]))
            gain1 = 0
            gain2 = 0
            for i in range(len(values) - 1):

                # Now we go to all various values of the feature. Create a threshold in the middle of the value and
                # the next value. Then split the set with this threshold and compute the probabilities for both
                # value in the sets. Then compute the chi-square value for both sets and if term in 269 is the maximum
                # till now we save it in our return values.
                threshold = (values[i + 1] + values[i]) / 2
                split1 = tmpy[tmpX[:, feature] <= threshold]
                split2 = tmpy[tmpX[:, feature] > threshold]
                if len(split1) == 0 or len(split2) == 0: continue
                expected = len(split1)/len(values)
                expected2 = len(split2) / len(values)
                for j in classes:
                    chi_square_of_d = math.sqrt(((len(split1[split1[:] == j]) -expected)**2)/expected)
                    chi_square_of_d2 = math.sqrt(((len(split2[split2[:] == j]) - expected2) ** 2) / expected2)
                    gain1 += chi_square_of_d 
                    gain2 += chi_square_of_d2
                chi_value = gain1 + gain2
                if chi_value > maxChi:
                    maxfeature = feature
                    maxthreshold = threshold
                    maxChi = chi_value
                    minchi1 = gain1
                    minchi2 = gain2

        return [maxfeature, maxthreshold, minchi1, minchi2]

    def __splitCART(self, X, y, gini):
        '''
            Here we check the max_features features to get the best split above them according to the gini value.
        :param X: Set to split.
        :param y: Classes of the set.
        :param gini: The value of the set.
        :return: Give a feature, threshold and the two entropie values of the split.
        '''

        # First we check how much features we should check with the the feature weights.
        numberFeatures = len(X[0])
        if self.max_features == 'sqrt':
            n = math.ceil(math.sqrt(numberFeatures))
            choice = np.random.choice([i for i in range(numberFeatures)], size=n, replace=False, p=self.weight_features)
        elif self.max_features == 'all':
            n = numberFeatures
            choice = [i for i in range(numberFeatures)]
        elif self.max_features == 'log':
            n = math.ceil(math.log2(numberFeatures))
            choice = np.random.choice([i for i in range(numberFeatures)], size=n, replace=False, p=self.weight_features)

        # Now we initialize our return variable.
        classes = list(set(y))
        mingini = gini
        minfeature = None
        minthreshold = None
        mingini1 = None
        mingini2 = None
        for feature in choice:

            # For each feature, we should check, we sort the values in the column of the feature and
            # create a set with all values but without double elements.
            tmpCols = list(X[:, feature])
            tmpCols.sort()
            itervalues = list(set(tmpCols))
            values = itervalues
            tmpy = y
            tmpX = X
            for i in range(len(values)-1):

                # Now we go to all various values of the feature. Create a threshold in the middle of the value and
                # the next value. Then split the set with this threshold and compute the probabilities for both
                # value in the sets. Then compute the gini value for both sets and if term in 334 is the minimum
                # till now we save it in our return values.
                threshold = (values[i+1] + values[i])/2
                split1 = tmpy[tmpX[:,feature]<= threshold]
                split2 = tmpy[tmpX[:, feature] > threshold]
                if len(split1) == 0 or len(split2) == 0: continue
                probs1 = []
                probs2 = []
                for j in classes:
                    probs1.append(len(split1[split1[:] == j]) / len(split1))
                    probs2.append(len(split2[split2[:] == j]) / len(split2))

                gini1 = giniImpurity(probs1)
                gini2 = giniImpurity(probs2)
                ginivalue = len(split1)/len(tmpX) * gini1 + len(split2)/len(tmpX) * gini2
                if ginivalue < mingini:
                    minfeature = feature
                    minthreshold = threshold
                    mingini = ginivalue
                    mingini1 = gini1
                    mingini2 = gini2

        return [minfeature,minthreshold,mingini1,mingini2]

if __name__ == '__main__':
    '''
    Test script to check the different functions of the class decision tree.
    '''
    from chooseData import datasets
    name = 'car'
    func = datasets(name)
    number = 100
    print(f'dataset: {name}')
    [train, y_train, test, y_test] = func()
    tree = Decisiontree()
    tree.fit(train.to_numpy(),y_train.to_numpy())

    y_pred = tree.predict(test.to_numpy())

    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_pred,y_test))
