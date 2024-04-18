import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

dataPerTrain = np.genfromtxt('perceptron-train.csv', delimiter=',')
dataPerTest = np.genfromtxt('perceptron-test.csv', delimiter=',')

Y_PerTrain = dataPerTrain[:,0]
X_PerTrain = dataPerTrain[:,1:]

Y_PerTest = dataPerTest[:,0]
X_PerTest = dataPerTest[:,1:]

clf = Perceptron( max_iter=5, tol=None, random_state=241)
perc =clf.fit(X_PerTrain,Y_PerTrain)
result = perc.predict(X_PerTest)
resultScore = accuracy_score(Y_PerTest, result)
print(resultScore)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_PerTrain)
X_test_scaled = scaler.transform(X_PerTest)

print('**************************')

clf2 = Perceptron( max_iter=5, tol=None, random_state=241)
perc2 =clf2.fit(X_train_scaled,Y_PerTrain)
result2 = perc2.predict(X_test_scaled)
resultScore2 = accuracy_score(Y_PerTest, result2)
print(resultScore2)