from sklearn.neighbors import KNeighborsClassifier
import sklearn.datasets
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold, cross_val_score

import numpy


# Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston()
boston = sklearn.datasets.load_boston()
# Приведите признаки в выборке к одному масштабу
boston.data = scale(boston.data)

# Создается генератор разбиений sklearn.cross_validation.KFold
# Качество оценивайте с помощью кросс-валидации по 5 блокам с random_state = 42, не забудьте включить
# перемешивание выборки (shuffle=True)
k_fold = KFold(len(boston.data), n_folds=5, shuffle=True, random_state=42)

quality = {}

# Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом, чтобы всего было протестировано 200 вариантов
for power in numpy.linspace(1, 10, num=200):
    # Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance'
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance', p=power, metric='minkowski')
    #quality[p] = max(cross_val_score(knn, X=boston.data, y=boston.target, cv=k_fold, scoring='mean_squared_error'))
    # В качестве метрики качества используйте среднеквадратичную ошибку (параметр scoring='mean_squared_error' у cross_val_score)
    quality[power] = cross_val_score(knn, X=boston.data, y=boston.target, cv=k_fold, scoring='mean_squared_error')

m = -1000
id = 0
for k, v in quality.items():
    print("%s => %s" % (k, v))
    if max(v) > m:
        m = max(v)
        id = k

#print(quality)
#print(max(quality.values()))
print("=======================================")
print("%s => %s" % (id, m))

