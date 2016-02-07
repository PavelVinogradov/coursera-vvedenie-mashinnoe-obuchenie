from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold, cross_val_score

import csv


def middle(values):
    sum = 0
    for value in values:
        sum += value

    return sum / len(values)


def knn(values, targets):

    # Создается генератор разбиений sklearn.cross_validation.KFold
    k_fold = KFold(len(wine_data), n_folds=5, shuffle=True, random_state=42)

    quality = {}

    # Вычисляется качество на всех разбиениях можно при помощи функции sklearn.cross_validation.cross_val_score
    for k in range(1, 51):
        knn = KNeighborsClassifier(n_neighbors=k)
        quality[k] = cross_val_score(knn, X=values, y=targets, cv=k_fold, scoring='accuracy')

    max = 0
    hi = 0
    for k, v in quality.items():
        if middle(v) > max:
            max = middle(v)
            hi = k

        #print('%s => %s => %s' % (k, v, middle(v)))

    print("%s => %s => %s" % (hi, quality[hi], middle(quality[hi])))


wine_data = []
wine_data_x = []
wine_data_y = []

with open('data/wine.data') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        wine_data_y.append(row[0])
        wine_data_x.append(row[1:])
        wine_data.append(row)

knn(wine_data_x, wine_data_y)

print("===========================================")

wine_data_x_scaled = scale(wine_data_x)

knn(wine_data_x_scaled, wine_data_y)