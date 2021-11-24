import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

''' 
Here are the methods to import the csv data as pandas dataframe. Each data set has its own method. 
The categorical data is automatically converted to numerical data and the data set is divided into training and test set.
Each method returns the training and test set separately in data and classes and a list of categorical feature.
'''

# Here we get the exact path of this script to be able to load the data from it.
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []

def arcene():
    '''
    This method handles the arcene data set. But it is to big for github.
    :return:
    train: pd.Dataframe of the training set of arcene.
    y_train: pd.Series of the classes of the train set of arcene.
    test: pd.Dataframe of the test set of arcene.
    y_test: pd.Series of the classes of the test set of arcene.
    CATEGORICAL_COLS: empty list because we have no categorical features in arcene.
    '''

    CATEGORICAL_COLS = []
    train = pd.read_csv(user_paths[0]+'/'+'arcene/arcene_train.data',sep=' ',header=None)
    test = pd.read_csv(user_paths[0] + '/' + 'arcene/arcene_valid.data', sep=' ', header=None)
    y_train = pd.read_csv(user_paths[0]+'/'+'arcene/arcene_train.labels',sep=' ',header=None)
    y_test = pd.read_csv(user_paths[0] + '/' + 'arcene/arcene_valid.labels', sep=' ', header=None)
    train[10000] = y_train[0]
    test[10000] = y_test[0]
    y_test = test.pop(10000)
    y_train = train.pop(10000)
    return [train, y_train, test, y_test,CATEGORICAL_COLS]

def amazon():
    '''
    This method handles the amazon data set. But it is to big for github.
    :return:
    train: pd.Dataframe of the training set of amazon.
    y_train: pd.Series of the classes of the train set of amazon.
    test: pd.Dataframe of the test set of amazon.
    y_test: pd.Series of the classes of the test set of amazon.
    CATEGORICAL_COLS: empty list because we have no categorical features in amazon.
    '''

    # Define the class feature and categorical features and a extra libary.
    from scipy.io import arff
    TARGET_COLUMN = ''
    CATEGORICAL_COLS = []

    # Load the data set
    PATH = user_paths[0]+'/'+'amazon/Amazon_initial_50_30_10000.arff'
    raw_data = arff.loadarff(PATH)
    data = pd.DataFrame(raw_data[0])

    # Splitting the data set in train and test set.
    msk = np.random.rand(len(data)) < 0.8
    y = data.pop(TARGET_COLUMN)
    train = data[msk]
    y_train = y[msk]
    test = data[~msk]
    y_test = y[~msk]
    return [train, y_train, test, y_test,CATEGORICAL_COLS]

def aps():
    '''
    This method handles the aps data set.
    :return:
    train: pd.Dataframe of the training set of aps.
    y_train: pd.Series of the classes of the train set of aps.
    test: pd.Dataframe of the test set of aps.
    y_test: pd.Series of the classes of the test set of aps.
    CATEGORICAL_COLS: empty list because we have no categorical features in aps.
    '''

    # Define the class feature and categorical features.
    TARGET_COLUMN = 'class'
    CATEGORICAL_COLS = []

    # Load the data set already in training and test set
    train = pd.read_csv(user_paths[0] + '/' + 'aps/aps_failure_training_set.csv', sep=',')#, header=None)
    test = pd.read_csv(user_paths[0] + '/' + 'aps/aps_failure_test_set.csv', sep=',')#, header=None)

    # Transform the categorical feature
    train = train.replace('na',-1)
    train[train[TARGET_COLUMN] == 'neg'] = 0
    train[train[TARGET_COLUMN] == 'pos'] = 1
    test = test.replace('na', -1)
    test[test[TARGET_COLUMN] == 'neg'] = 0
    test[test[TARGET_COLUMN] == 'pos'] = 1
    y_train = pd.to_numeric(train.pop(TARGET_COLUMN))
    y_test = pd.to_numeric(test.pop(TARGET_COLUMN))
    test = test.apply(pd.to_numeric)
    train = train.apply(pd.to_numeric)
    test = test.astype(np.float32)
    train = train.astype(np.float32)
    return [train, y_train, test, y_test,CATEGORICAL_COLS]

def avila():
    '''
    This method handles the avila data set.
    :return:
    train: pd.Dataframe of the training set of avila.
    y_train: pd.Series of the classes of the train set of avila.
    test: pd.Dataframe of the test set of avila.
    y_test: pd.Series of the classes of the test set of avila.
    CATEGORICAL_COLS: empty list because we have no categorical features in avila.
    '''

    # Define the class feature and categorical features.
    CATEGORICAL_COLS = []
    TARGET_COLUMN = 'output' #'class'
    AVILA_COLUMNS = [f'F{i+1}' for i in range(10)]
    AVILA_COLUMNS.append('output')

    # Load the data sets.
    data1 = pd.read_csv(user_paths[0]+'/'+'avila/avila-tr.txt',header=None,names=AVILA_COLUMNS)
    data2 = pd.read_csv(user_paths[0]+'/'+'avila/avila-ts.txt',header=None,names=AVILA_COLUMNS)
    data = data1.append(data2)

    # Transform the categorical feature into numerical one
    le = LabelEncoder()
    le.fit(data[TARGET_COLUMN])
    data[TARGET_COLUMN] = le.transform(data[TARGET_COLUMN])

    # Splitting the data set in train and test set.
    n1 = len(data1)
    n2 = len(data2)
    msk = np.array([i < n1 for i in range(n1+n2)])
    y = data.pop(TARGET_COLUMN)
    train = data[msk]
    y_train = y[msk]
    test = data[~msk]
    y_test = y[~msk]

    return [train,y_train,test,y_test,CATEGORICAL_COLS]

def breast():
    '''
    This method handles the breast cancer data set.
    :return:
    train: pd.Dataframe of the training set of breast.
    y_train: pd.Series of the classes of the train set of breast.
    test: pd.Dataframe of the test set of breast.
    y_test: pd.Series of the classes of the test set of breast.
    CATEGORICAL_COLS: list of the features that are categorical.
    '''

    # Define the class feature and categorical features.
    TARGET_COLUMN = 'Class'
    PATH = user_paths[0] + '/' + 'breast/breast-cancer-wisconsin.data'
    CATEGORICAL_COLS = ['Clump Thickness', 'Uniformity od Cell Size', 'Uniformity od Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    COLS = ['ID', 'Clump Thickness', 'Uniformity od Cell Size', 'Uniformity od Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    # Load the data set.
    data = pd.read_csv(PATH, sep=',', names=COLS, header=None)  #  index_col=0,

    # Delete the ID column because there are no relevant information in it.
    data.pop('ID')
    arr_delete = []
    for key, item in data.iterrows():
        for col in data.columns:
            if item[col] == '?':
                if not key in arr_delete:
                    arr_delete.append(key)
    data = data.drop(arr_delete)

    # Transform the categorical feature.
    for col in data.columns:
        data[col] = pd.to_numeric(data[col])

    # Splitting the data set in train and test set.
    msk = np.random.rand(len(data)) < 0.8
    y = data.pop(TARGET_COLUMN)
    train = data[msk]
    y_train = y[msk]
    test = data[~msk]
    y_test = y[~msk]
    return [train, y_train, test, y_test, CATEGORICAL_COLS]

def car():
    '''
    This method handles the car data set.
    :return:
    train: pd.Dataframe of the training set of car.
    y_train: pd.Series of the classes of the train set of car.
    test: pd.Dataframe of the test set of car.
    y_test: pd.Series of the classes of the test set of car.
    CATEGORICAL_COLS: list of the features that are categorical.
    '''

    # Define the class feature and categorical features.
    TARGET_COLUMN = 'output'
    CAR_COLUMNS = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'output']
    CATEGORICAL_COLUMNS = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

    # Load the data set
    data = pd.read_csv(user_paths[0]+'/'+'car/car.data',header=None,names=CAR_COLUMNS)

    # Transform the categorical feature
    for item in ['buying', 'maint','doors', 'persons', 'lug_boot', 'safety','output']:
        le = LabelEncoder()
        le.fit(data[item])
        data[item] = le.transform(data[item])

    # Splitting the data set in train and test set.
    msk = np.random.rand(len(data)) < 0.8
    y = data.pop(TARGET_COLUMN)
    train = data[msk]
    y_train = y[msk]
    test = data[~msk]
    y_test = y[~msk]
    return [train, y_train, test, y_test, CATEGORICAL_COLUMNS]

def creditcard():
    '''
    This method handles the creditcard data set.
    :return:
    train: pd.Dataframe of the training set of creditcard.
    y_train: pd.Series of the classes of the train set of creditcard.
    test: pd.Dataframe of the test set of creditcard.
    y_test: pd.Series of the classes of the test set of creditcard.
    CATEGORICAL_COLS: list of the features that are categorical.
    '''

    # Define the class feature and categorical features.
    TARGET_COLUMN = 'default.payment.next.month'
    CATEGORICAL_COLUMNS = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    # Load the data set.
    data =pd.read_csv(user_paths[0]+'/'+'creditcard/UCI_Credit_Card.csv',sep=',')

    # Delete the ID column because there are no relevant information in it.
    data.pop('ID')

    # Splitting the data set in train and test set.
    n = len(data)
    msk = np.random.rand(n) < 0.8
    y = data.pop(TARGET_COLUMN)
    train = data[msk]
    y_train = y[msk]
    test = data[~msk]
    y_test = y[~msk]
    return [train, y_train, test, y_test,CATEGORICAL_COLUMNS]

def elec():
    '''
    This method handles the electric data set.
    :return:
    train: pd.Dataframe of the training set of electric.
    y_train: pd.Series of the classes of the train set of electric.
    test: pd.Dataframe of the test set of electric.
    y_test: pd.Series of the classes of the test set of electric.
    CATEGORICAL_COLS: empty list because we have no categorical features in electric.
    '''

    # Define the class feature and categorical features.
    TARGET_COLUMN = 'stabf'
    CATEGORICAL_COLS = []

    # Load the data set    .
    DATA_FILE = user_paths[0]+'/'+'electric/Data_for_UCI_named.csv'
    data = pd.read_csv(DATA_FILE)

    # Transform the categorical feature into numerical one.
    le = LabelEncoder()
    le.fit(data[TARGET_COLUMN])
    data[TARGET_COLUMN] = le.transform(data[TARGET_COLUMN])

    # Splitting the data set in train and test set.
    n = len(data)
    msk = np.random.rand(n) < 0.8
    y = data.pop(TARGET_COLUMN)
    train = data[msk]
    y_train = y[msk]
    test = data[~msk]
    y_test = y[~msk]
    return [train, y_train, test, y_test,CATEGORICAL_COLS]

def generator(max_number_features=100):
    '''

    :param max_number_features:
    :return:
    train: pd.Dataframe of the training set of generator.
    y_train: pd.Series of the classes of the train set of generator.
    test: pd.Dataframe of the test set of generator.
    y_test: pd.Series of the classes of the test set of generator.
    CATEGORICAL_COLS: empty list because we have no categorical features in generator.
    '''

    # Define the categorical features.
    CATEGORICAL_COLUMNS = []

    # Create the data set with the fix parameter and the parameter max_number_features.
    X, y = make_classification(n_samples=300, n_features=max_number_features,n_informative=max_number_features,n_redundant=0)

    # Splitting the data set in train and test set.
    train, test, y_train, y_test = train_test_split(X, y)
    train = pd.DataFrame(train, columns=[str(i) for i in range(len(X[0]))])
    test = pd.DataFrame(test, columns=[str(i) for i in range(len(X[0]))])
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    return [train, y_train, test, y_test,CATEGORICAL_COLUMNS]

def glass():
    '''
    This method handles the glass data set.
    :return:
    train: pd.Dataframe of the training set of glass.
    y_train: pd.Series of the classes of the train set of glass.
    test: pd.Dataframe of the test set of glass.
    y_test: pd.Series of the classes of the test set of glass.
    CATEGORICAL_COLS: empty list because we have no categorical features in glass.
    '''

    # Define the class feature and categorical features.
    CATEGORICAL_COLS = []
    TARGET_COLUMN = 'class'
    PATH = user_paths[0] + '/' + 'glass/glass.data'
    COLS = ['ID','RI','Na','MG','Al','Si','K','Ca','Ba','Fe','class']

    # Load the data set and use the first col as index.
    data = pd.read_csv(PATH,sep=',',index_col=0,names=COLS,header=None)

    # Splitting the data set in train and test set.
    msk = np.random.rand(len(data)) < 0.8
    y = data.pop(TARGET_COLUMN)
    train = data[msk]
    y_train = y[msk]
    test = data[~msk]
    y_test = y[~msk]
    return [train,y_train,test,y_test,CATEGORICAL_COLS]

def heart():
    '''
    This method handles the heart data set.
    :return:
    train: pd.Dataframe of the training set of heart.
    y_train: pd.Series of the classes of the train set of heart.
    test: pd.Dataframe of the test set of heart.
    y_test: pd.Series of the classes of the test set of heart.
    CATEGORICAL_COLS: list of the features that are categorical.
    '''

    # Define the class feature and categorical features.
    CATEGORICAL_COLUMNS = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
    TARGET_COLUMN = 'output'

    # Load the data set
    DATA_FILE = user_paths[0] + '/' + 'heart/heart.csv'
    data = pd.read_csv(DATA_FILE)

    # Splitting the data set in train and test set.
    n = len(data)
    msk = np.random.rand(n) < 0.8
    y = data.pop(TARGET_COLUMN)
    train = data[msk]
    y_train = y[msk]
    test = data[~msk]
    y_test = y[~msk]
    return [train, y_train, test, y_test, CATEGORICAL_COLUMNS]

def isolet():
    '''
    This method handles the isolet data set.
    :return:
    train: pd.Dataframe of the training set of isolet.
    y_train: pd.Series of the classes of the train set of isolet.
    test: pd.Dataframe of the test set of isolet.
    y_test: pd.Series of the classes of the test set of isolet.
    CATEGORICAL_COLS: list of the features that are categorical.
    '''

    # Define the class feature and categorical features.
    TARGET_COLUMN = 617
    CATEGORICAL_COLS = ['isolet']

    # Load the data sets
    data1 = pd.read_csv(user_paths[0] + '/' + 'isolet/isolet1+2+3+4.data', header=None, sep=',')
    data2 = pd.read_csv(user_paths[0] + '/' + 'isolet/isolet5.data', header=None, sep=',')


    # Create a feature for the data set depending on which file they are in.
    data1['isolet'] = 1.0
    data2['isolet'] = 5.0
    data = data1.append(data2)

    # Splitting the data set in train and test set.
    msk = np.random.rand(len(data)) < 0.8
    y = data.pop(TARGET_COLUMN)
    train = data[msk]
    y_train = y[msk]
    test = data[~msk]
    y_test = y[~msk]
    test = test.astype(np.float32)
    train = train.astype(np.float32)
    return [train, y_train, test, y_test,CATEGORICAL_COLS]

def mushrooms():
    '''
    This method handles the mushrooms data set.
    :return:
    train: pd.Dataframe of the training set of mushrooms.
    y_train: pd.Series of the classes of the train set of mushrooms.
    test: pd.Dataframe of the test set of mushrooms.
    y_test: pd.Series of the classes of the test set of mushrooms.
    categorical_columns: list of the features that are categorical.
    '''

    # Define the class feature.
    TARGET_COLUMN = 'class'

    # Load the data set
    data = pd.read_csv(user_paths[0]+'/'+'mushrooms/mushrooms.csv', sep=',')

    # Transform the categorical feature into numerical one.
    for item in data.columns:
        le = LabelEncoder()
        le.fit(data[item])
        data[item] = le.transform(data[item])

    # Splitting the data set in train and test set.
    n = len(data)
    msk = np.random.rand(n) < 0.8
    y = data.pop(TARGET_COLUMN)
    train = data[msk]
    y_train = y[msk]
    test = data[~msk]
    y_test = y[~msk]

    # All features are categorical so we put them into the list
    categorical_columns = data.columns
    return [train, y_train, test, y_test,categorical_columns]

def wine():
    '''
    This method handles the wine data set.
    :return:
    train: pd.Dataframe of the training set of wine.
    y_train: pd.Series of the classes of the train set of wine.
    test: pd.Dataframe of the test set of wine.
    y_test: pd.Series of the classes of the test set of wine.
    CATEGORICAL_COLS: list of the features that are categorical.
    '''

    # Define the class feature and categorical features.
    TARGET_COLUMN = 'quality'
    CATEGORICAL_COLUMNS = ['color']

    # Load the data sets.
    data1 =pd.read_csv(user_paths[0]+'/'+'wine/winequality-red.csv',sep=';')
    data2 = pd.read_csv(user_paths[0]+'/'+'wine/winequality-white.csv', sep=';')

    # Create a feature for the data set depending on which file they are in. This is the color of the wine.
    data1['color'] = 0
    data2['color'] = 1
    data = data1.append(data2)

    # Splitting the data set in train and test set.
    n = len(data)
    msk = np.random.rand(n) < 0.8
    y = data.pop(TARGET_COLUMN)
    train = data[msk]
    y_train = y[msk]
    test = data[~msk]
    y_test = y[~msk]
    return [train, y_train, test, y_test,CATEGORICAL_COLUMNS]

# A public dictonary from all the above methods about it.
DATASETS = {'breast':breast,'glass':glass,'elec':elec,'aps': aps,  'isolet': isolet,  'car': car, 'avila': avila,
                'heart': heart,'heart_test': heart, 'wine': wine, 'mushrooms': mushrooms, 'creditcard': creditcard,'generator':generator}
def datasets(name):
    '''
        This function only gives the right method back.
    :param name: string with the name of the data set.
    :return: function handle of the method to load the data set.
    '''
    return DATASETS[name]
