import math
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

'''
In this file we store our different measures and method to compute all 
direct measures and all compare measures with one call. 
'''

def onceMeasures(y_true,y_classifier):
    '''
        Here we use the different measures methof of sklearn to compute them and give them as list back.
    :param y_true: A list of all true values of the class.
    :param y_classifier: A list of all predicted values of the class.
    :return:
    acc: The accuracy measures value.
    f1_micro: The f1 score of micro method.
    f1_macro: The f1 score of macro method.
    confus: The confusion matrix of the both lists.
    '''
    acc = accuracy_score(y_true,y_classifier)
    f1_micro = f1_score(y_true,y_classifier,average='micro')
    f1_macro = f1_score(y_true, y_classifier, average='macro')
    confus = confusion_matrix(y_true,y_classifier)
    return [acc,f1_micro,f1_macro,confus]


def compareMeasures(y_true, y_classifier1, y_classifier2):
    '''
        We compute the matrix of the section 4.2 compare measures of the masterthesis.
        After that we calculate the McNemar coefficient and the coefficients of the confidence interval.
    :param y_true: A list of all true values of the class.
    :param y_classifier1: A list of all predicted values of the class from a classifier.
    :param y_classifier2: A list of all predicted values of the class from a classifier.
    :return:
    The mcnemar coeffient and the coefficients of the confidence interval.
    '''

    #Here we compute the matrix of the section 4.2 compare measures of the masterthesis.
    n_true = 0
    n01 = 0
    n10 = 0
    n_false = 0
    n = len(y_true)
    for i in range(n):
        if not y_classifier1[i] == y_classifier2[i]:
            if y_classifier1[i] == y_true[i]:
                n01 += 1
            if y_classifier2[i] == y_true[i]:
                n10 += 1
        else:
            if y_classifier1[i] == y_true[i]:
                n_true +=1
            else:
                n_false +=1
    # Call the methods mcnemar and __confidenceInterval
    return [__mcnemar(n01, n10), __confidenceInterval(n, n01, n10, n_false)]


def __mcnemar(n01, n10):
    '''
        Privat method to compute the McNemar coefficient.
    :param n01: The number of examples correctly classified by one of the classifierand misclassified by
    the other classifier.
    :param n10: vice versa.
    :return:
    the mcnemar coefficient.
    '''
    tmp1 = (abs(n01 - n10) - 1) ** 2
    tmp2 = n01 + n10
    if tmp2 != 0:
        mcnemar = (tmp1 /tmp2)
    else:
        mcnemar = 0
    return mcnemar

def __confidenceInterval(n, n01, n10, n_false):
    '''
    Privat method to compute the coefficients of the confidence interval.
    :param n: Total number of elements in the predicted classes.
    :param n01: The number of examples correctly classified by one of the classifierand misclassified by
    the other classifier.
    :param n10: vice versa.
    :param n_false: The number of examples both classifiers misclassified.
    :return:
    The coefficients of the confidence interval.
    '''
    p01 = n01 / n
    p10 = n10 / n
    p_false = n_false/n
    tmp1 = (p01-p10)**2
    tmp1 += p01+p10
    tmp1 = tmp1/n
    SE = math.sqrt(tmp1)
    pA = p01 +p_false
    pB = p10 +p_false
    return [SE,pA,pB]