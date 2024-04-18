import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import KFold

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )
news = newsgroups.data
Y_news = newsgroups.target
vectorizer = TfidfVectorizer()
X_news = vectorizer.fit_transform(news)


cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
grid = {'C': np.power(10.0, np.arange(-5, 6))}
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X_news,Y_news)
