import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
data = np.genfromtxt('wine.data', delimiter=',')
y = data[:,0]
x = data[:,1:]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
array50 = np.arange(51)
for k_number in array50:
    knn = KNeighborsClassifier(n_neighbors=k_number)   
    vals = cross_val_score(knn, x, y, cv=kf, scoring='accuracy')
    print('число соседей:', k_number, ' среднее:',np.mean(vals))   
  
print('нормализация *****************************')
X_scaled = preprocessing.scale(x)
for k_number in array50:
    knn = KNeighborsClassifier(n_neighbors=k_number)   
    vals = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
    print('число соседей:', k_number,' среднее:', np.mean(vals))


from sklearn.datasets import load_boston
from sklearn import preprocessing
X, y = load_boston(return_X_y=True)
coor = preprocessing.scale(X)
array10 = np.arange(11)
for k_number in array10:
neigh = KNeighborsRegressor(n_neighbors=2, )



result = prec.predict(X_PerTest)