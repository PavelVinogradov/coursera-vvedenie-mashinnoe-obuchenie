import pandas as pd
import math
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split

#TODO: sync with https://github.com/Dreamastiy/IntroToMachineLearningHSEYandex/blob/master/BiologicalResponse.ipynb

#1
# Загрузите выборку из файла gbm-data.csv с помощью pandas
#
# В первой колонке файла с данными записано, была или нет реакция.
# Все остальные колонки (d1 - d1776) содержат различные характеристики молекулы, такие как размер, форма и т.д.

data = pd.read_csv('data/gbm-data.csv')

# Преобразуйте ее в массив numpy (параметр values у датафрейма).
data_values = data.values
X = data_values[:, 1:]
y = data_values[:, 0]


# Разбейте выборку на обучающую и тестовую, используя функцию train_test_split
# с параметрами test_size = 0.8 и random_state = 241.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

#2
# Обучите GradientBoostingClassifier с параметрами n_estimators=250, verbose=True, random_state=241
# и для каждого значения learning_rate из списка [1, 0.5, 0.3, 0.2, 0.1] проделайте следующее:
for lr in [1, 0.5, 0.3, 0.2, 0.1]:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=lr)
    clf.fit(X_train, y_train)

    # Используйте метод staged_decision_function для предсказания качества на обучающей и тестовой выборке на каждой итерации.
    score_prediction_train = clf.staged_decision_function(X_train)
    score_prediction_test = clf.staged_decision_function(X_test)

    # Преобразуйте полученное предсказание с помощью сигмоидной функции по формуле 1 / (1 + e^{−y_pred}), где y_pred — предсказаное значение.
    score_prediction_train_mod = (1 / (1 + math.exp(-score_prediction_train)))
    score_prediction_test_mod = (1 / (1 + math.exp(-score_prediction_test)))

    # Вычислите и постройте график значений log-loss
    # (которую можно посчитать с помощью функции sklearn.metrics.log_loss) на обучающей и тестовой выборках,
    # а также найдите минимальное значение метрики и номер итерации, на которой оно достигается.
    log_loss_graph_train = log_loss(y_train, clf.predict_proba(X_train)[:, 1])
    log_loss_graph_test = log_loss(y_test, clf.predict_proba(X_test)[:, 1])

    print("%s -> ll[train] = %s -> ll[test] = %s" % (lr, log_loss_graph_train, log_loss_graph_test))

