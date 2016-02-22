import pandas as pd
from sklearn.decomposition import PCA
from numpy import corrcoef, cumsum

#1
prices_data = pd.read_csv('data/close_prices.csv')

print(prices_data.head())
#print(prices_data.loc[:,'AXP':].head())

#2
X = prices_data.ix[:,1:]
pca = PCA(n_components=10)
clf = pca.fit(X)

print("explained_variance_ratio_ => %s => %s" % (clf.explained_variance_ratio_, cumsum(clf.explained_variance_ratio_)))

#3
t = clf.transform(X)
#print("components_ => %s" % clf.components_)

first_component = t[:, 0]
#print("[%s] : %s" % (len(first_component), first_component))
print("[%s]" % (len(first_component)))

#4
djia_data = pd.read_csv('data/djia_index.csv')
print(djia_data.head())
#print("[%s] : %s" % (len(djia_data['^DJI']), djia_data['^DJI']))
print("[%s]" % (len(djia_data['^DJI'])))

correlation = corrcoef(first_component, djia_data['^DJI'])
print("corrcoef => %s" % correlation)

print(max(clf.components_[0]))
print(clf.components_[0])