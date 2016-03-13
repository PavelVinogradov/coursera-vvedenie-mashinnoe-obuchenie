import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold, cross_val_score


def r2_scorer(estimator, X, y):
    predictions = estimator.predict(X)
    return r2_score(y, predictions)

#1
# Загрузите данные из файла abalone.csv.
# Это датасет, в котором требуется предсказать возраст ракушки (число колец) по физическим измерениям.
data = pd.read_csv('data/abalone.csv')

#2
# Преобразуйте признак Sex в числовой: значение F должно перейти в -1, I — в 0, M — в 1.
# Если вы используете Pandas, то подойдет следующий код:
#   data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

#3
# Разделите содержимое файлов на признаки и целевую переменную. В последнем столбце записана целевая переменная, в остальных — признаки.
X = data.ix[:, :-1]
y = data.ix[:, -1]

#4
# Обучите случайный лес (sklearn.ensemble.RandomForestRegressor) с различным числом деревьев: от 1 до 50
# (не забудьте выставить "random_state=1" в конструкторе).

# Используйте параметры "random_state=1" и "shuffle=True" при создании генератора кросс-валидации
# sklearn.cross_validation.KFold.
k_fold = KFold(len(X), n_folds=5, shuffle=True, random_state=1)

for x in range(1, 50):
    clf = RandomForestRegressor(n_estimators=x, random_state=1)
    # clf.fit(X, y)

    # Для каждого из вариантов оцените качество работы полученного леса на кросс-валидации по 5 блокам.
    score = cross_val_score(estimator=clf, X=X, y=y, cv=k_fold, scoring=r2_scorer)

    #predictions = clf.predict(X)
    # В качестве меры качества воспользуйтесь долей правильных ответов (sklearn.metrics.r2_score).
    #score = r2_score(y, predictions)

    print("%s => %s => %s" % (x, score, sum(score) / len(score)))



#print(X.head())
#print(y.head())
#print(predictions[:5])
