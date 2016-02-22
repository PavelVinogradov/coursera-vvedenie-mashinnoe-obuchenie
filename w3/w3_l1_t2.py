from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
import numpy as np

# Load data
newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
transformer = TfidfVectorizer()

# Vectorize text
X_train_tfidf = transformer.fit_transform(newsgroups.data)
X_test_tfidf = transformer.transform(newsgroups.data, copy=True)

y = newsgroups.target
X = X_train_tfidf


# Tune C
def tune():
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(X, y)

    # TODO: Learn this hack
    best_parameter, score, _ = max(gs.grid_scores_, key=lambda x: x[1])
    print('C => %s' % best_parameter['C'])

    for a in gs.grid_scores_:
        print("%s => %s" % (a.parameters, a.mean_validation_score))

    return best_parameter['C']

#C = tune()
C = 1.0
clf = SVC(kernel='linear', random_state=241, C=C)
clf.fit(X, y)

temp = np.argsort(abs(clf.coef_.data))
words = temp[-10:]

features = transformer.get_feature_names()
temp = []
for word in words:
    temp.append((features[clf.coef_.indices[word]]))
temp.sort()
for x in temp:
    print(x)

#coef = clf.coef_
#s_coef = coef.sorted_indices()
#a_coef = coef.toarray()
#print(coef)

#print(newsgroups)