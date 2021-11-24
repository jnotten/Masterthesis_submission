import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from matplotlib import pyplot as plt



data = pd.read_csv('heart.csv')
n = len(data)

df = pd.DataFrame(np.random.randn(n, 2))
msk = np.random.rand(n) < 0.8

train = data[msk]
y_train = train.pop('output')
test = data[~msk]
y_test = test.pop('output')

x = [i for i in range(1,600,1)]
y1 = []
y2 = []
for i in x:
    number_trees = i

    clf = RandomForestClassifier(n_estimators=number_trees,n_jobs=-1)

    clf = clf.fit(train,y_train)


    # Evaluation.
    y_pred_test = clf.predict(test)
    y_pred_train = clf.predict(train)
    # View accuracy score

    y1.append(accuracy_score(y_test, y_pred_test))
    y2.append(mean_squared_error(y_test, y_pred_test))

    # print(f'________SK_RF_______number of tree: {i}_____')
    # print(f'accuracy test: {accuracy_score(y_test, y_pred_test)}')
    # print(f'accuracy train: {accuracy_score(y_train, y_pred_train)}')
    # print(f'mse test: {mean_squared_error(y_test, y_pred_test)}')
    #print(confusion_matrix(y_test,y_pred_test))
    #print(classification_report(y_test,y_pred_test))


plt.plot(x,y1)
plt.show()
plt.plot(x,y2)
plt.show()