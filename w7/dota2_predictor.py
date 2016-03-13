import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc

import numpy
import datetime


def boosting():
    # Подход 1: градиентный бустинг "в лоб"

    # Один из самых универсальных алгоритмов, изученных в нашем курсе, является градиентный бустинг. Он не очень
    # требователен к данным, восстанавливает нелинейные зависимости, и хорошо работает на многих наборах данных, что и
    # обуславливает его популярность. В данном разделе предлагается попробовать градиентный бустинг для решения нашей
    # задачи.

    # 1. Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше. Удалите признаки, связанные
    # с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).
    train_data = pd.read_csv('data/features_training.csv', index_col='match_id')
    X_train = train_data.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                           'barracks_status_radiant', 'barracks_status_dire'], axis=1)
    y_train = train_data['radiant_win']


    # 2. Какие признаки имеют пропуски среди своих значений (приведите полный список имен этих признаков)? Что могут
    # означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?
    train_data_is_null = train_data.isnull().sum()
    print('*************************************************')
    print('1. Признаки имеющие пропуски среди своих значений')
    print(train_data_is_null[train_data_is_null > 0])

    #Признаки события "первая кровь" (first blood). Если событие "первая кровь" не успело произойти за первые 5 минут, то признаки принимают пропущенное значение
    #   first_blood_time: игровое время первой крови
    #   first_blood_team: команда, совершившая первую кровь (0 — Radiant, 1 — Dire)

    # 3. Замените пропуски на нули с помощью функции fillna().
    X_train.fillna(value=0, inplace=True)

    # 4. Какой столбец содержит целевую переменную? Запишите его название.
    #print(train_data['radiant_win'])
    print('*************************************************')
    print('2. Какой столбец содержит целевую переменную: %s' % 'radiant_win')

    # 5. Забудем, что в выборке есть категориальные признаки, и попробуем обучить градиентный бустинг над деревьями на
    # имеющейся матрице "объекты-признаки". Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold), не
    # забудьте перемешать при этом выборку (shuffle=True), поскольку данные в таблице отсортированы по времени, и без
    # перемешивания можно столкнуться с нежелательными эффектами при оценивании качества.
    k_fold = KFold(len(X_train), n_folds=5, shuffle=True, random_state=1)

    # Оцените качество градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации, попробуйте при
    # этом разное количество деревьев (как минимум протестируйте следующие значения для количества деревьев: 10, 20, 30).
    # Долго ли настраивались классификаторы? Достигнут ли оптимум на испытанных значениях параметра n_estimators, или же
    # качество, скорее всего, продолжит расти при дальнейшем его увеличении?
    print('*************************************************')

    # 3. Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями?
    # Какое качество при этом получилось?
    for num_tree in [10, 20, 30]:
        clf = GradientBoostingClassifier(n_estimators=num_tree, random_state=241, verbose=0)

        start_time = datetime.datetime.now()
        score = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=k_fold, scoring='roc_auc')
        end_time = datetime.datetime.now()

        print("%s => \n\ttime to fit = %s\n\tROC_AUC = %s" %
              (num_tree, end_time-start_time, score.mean()))

    # 4. Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге?
    # Что можно сделать, чтобы ускорить его обучение при увеличении количества деревьев?

####################################################################################################

# Подход 2: логистическая регрессия
def logreg():
    # Считайте таблицу с признаками из файла features.csv.
    train_data = pd.read_csv('data/features_training.csv', index_col='match_id')
    X_train = train_data.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                           'barracks_status_radiant', 'barracks_status_dire'], axis=1)
    y_train = train_data['radiant_win']

    # Замените пропуски на нули с помощью функции fillna().
    X_train.fillna(value=0, inplace=True)


    # 1. Оцените качество логистической регрессии (sklearn.linear_model.LogisticRegression с L2-регуляризацией) с
    # помощью кросс-валидации по той же схеме, которая использовалась для градиентного бустинга. Подберите при этом
    # лучший параметр регуляризации (C). Какое наилучшее качество у вас получилось? Как оно соотносится с качеством
    # градиентного бустинга? Чем вы можете объяснить эту разницу? Быстрее ли работает логистическая регрессия по
    # сравнению с градиентным бустингом?

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))

    # Разобъем данные на тестовую и обучающую выборки
    X_small_train, X_small_test, y_small_train, y_small_test = train_test_split(X_train_scaled, y_train, test_size = 0.5,
                                                                                random_state = 1)

    # Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold), не
    # забудьте перемешать при этом выборку (shuffle=True), поскольку данные в таблице отсортированы по времени, и без
    # перемешивания можно столкнуться с нежелательными эффектами при оценивании качества.
    k_fold = KFold(len(X_small_train), n_folds=5, shuffle=True, random_state=1)

    # Проведем поиск по сетке параметров - в качестве параметра будет выступать коэффициент регуляризации 'C':
    logreg_params = {'C': numpy.power(10.0, numpy.arange(-5, 6, 1))}

    clf_logreg = LogisticRegression(random_state=1, verbose=0)
    clf_logreg_grid = GridSearchCV(clf_logreg,
                                   logreg_params,
                                   cv=k_fold,
                                   scoring='roc_auc'
                                   )
    clf_logreg_grid.fit(X_small_train, y_small_train)


    y_train_score_logreg = clf_logreg_grid.decision_function(X_small_train)
    y_test_score_logreg = clf_logreg_grid.decision_function(X_small_test)

    # Какое качество получилось у логистической регрессии над всеми исходными признаками?
    # Как оно соотносится с качеством градиентного бустинга? Чем можно объяснить эту разницу?
    # Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?
    print(clf_logreg_grid.best_params_)
    print(roc_auc_score(y_small_train, y_train_score_logreg))
    print(roc_auc_score(y_small_test, y_test_score_logreg))

    ###################################################################################################################

    # 2. Среди признаков в выборке есть категориальные, которые мы использовали как числовые, что вряд ли является
    # хорошей идеей. Категориальных признаков в этой задаче одиннадцать: lobby_type и r1_hero, r2_hero, ..., r5_hero,
    # d1_hero, d2_hero, ..., d5_hero. Уберите их из выборки, и проведите кросс-валидацию для логистической регрессии
    # на новой выборке с подбором лучшего параметра регуляризации. Изменилось ли качество? Чем вы можете это объяснить?

    X_train_cleaned = X_train.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                                    'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1)

    # масштабируем признаки
    scaler = StandardScaler()
    scaler.fit(X_train_cleaned)
    X_train_cleaned_scaled = pd.DataFrame(scaler.transform(X_train_cleaned))

    # Разобъем данные на тестовую и обучающую выборки
    X_small_train, X_small_test, y_small_train, y_small_test = train_test_split(X_train_cleaned_scaled, y_train,
                                                                                test_size=0.5, random_state=1)

    k_fold = KFold(len(X_small_train), n_folds=5, shuffle=True, random_state=1)

    # Проведем поиск по сетке параметров - в качестве параметра будет выступать коэффициент регуляризации 'C':
    logreg_params = {'C': numpy.power(10.0, numpy.arange(-5, 6, 1))}

    clf_logreg = LogisticRegression(random_state=1, verbose=0)
    clf_logreg_grid = GridSearchCV(clf_logreg,
                                   logreg_params,
                                   cv=k_fold,
                                   scoring='roc_auc'
                                   )

    clf_logreg_grid.fit(X_small_train, y_small_train)

    y_train_score_logreg = clf_logreg_grid.decision_function(X_small_train)
    y_test_score_logreg = clf_logreg_grid.decision_function(X_small_test)

    # Какое качество получилось у логистической регрессии над некатегориальными исходными признаками?
    print(clf_logreg_grid.best_params_)
    print(roc_auc_score(y_small_train, y_train_score_logreg))
    print(roc_auc_score(y_small_test, y_test_score_logreg))

    ###################################################################################################################

    # 3. На предыдущем шаге мы исключили из выборки признаки rM_hero и dM_hero, которые показывают, какие именно герои
    # играли за каждую команду. Это важные признаки — герои имеют разные характеристики, и некоторые из них
    # выигрывают чаще, чем другие. Выясните из данных, сколько различных идентификаторов героев существует в данной
    # игре (вам может пригодиться фукнция unique или value_counts).

    heroes = train_data[['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                                    'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']]

    heroes = heroes.stack().unique()
    #print(heroes.head())
    print(len(heroes))

    ###################################################################################################################

    # 4. Воспользуемся подходом "мешок слов" для кодирования информации о героях. Пусть всего в игре имеет N различных
    # героев. Сформируем N признаков, при этом i-й будет равен
    #   нулю, если i-й герой не участвовал в матче;
    #   единице, если i-й герой играл за команду Radiant;
    #   минус единице, если i-й герой играл за команду Dire.

    # N — количество различных героев в выборке
    N = heroes.max()
    X_pick = numpy.zeros((X_train.shape[0], N))

    for i, match_id in enumerate(X_train.index):
        for p in range(5):
            X_pick[i, X_train.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, X_train.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

    #print(X_pick)

    X_pick_df = pd.DataFrame(X_pick)
    X_pick_df.columns = range(1, N + 1)
    cols = [col for col in X_pick_df.columns if col in heroes]
    X_pick_df = X_pick_df[cols]

    X_fin = pd.concat([X_train_cleaned_scaled, X_pick_df], axis = 1)

    #print(X_fin.head())

    ###################################################################################################################

    # 5. Проведите кросс-валидацию для логистической регрессии на новой выборке с подбором лучшего параметра
    # регуляризации. Какое получилось качество? Улучшилось ли оно? Чем вы можете это объяснить?

    # Разобъем данные на тестовую и обучающую выборки
    X_small_train, X_small_test, y_small_train, y_small_test = train_test_split(X_fin, y_train,
                                                                                test_size=0.5, random_state=1)

    k_fold = KFold(len(X_small_train), n_folds=5, shuffle=True, random_state=1)

    # Проведем поиск по сетке параметров - в качестве параметра будет выступать коэффициент регуляризации 'C':
    logreg_params = {'C': numpy.power(10.0, numpy.arange(-5, 6, 1))}

    clf_logreg = LogisticRegression(random_state=1, verbose=0)
    clf_logreg_grid = GridSearchCV(clf_logreg,
                                   logreg_params,
                                   cv=k_fold,
                                   scoring='roc_auc'
                                   )

    clf_logreg_grid.fit(X_small_train, y_small_train)

    y_train_score_logreg = clf_logreg_grid.decision_function(X_small_train)
    y_test_score_logreg = clf_logreg_grid.decision_function(X_small_test)

    # Какое качество получилось у логистической регрессии над некатегориальными исходными признаками?
    print(clf_logreg_grid.best_params_)
    print(roc_auc_score(y_small_train, y_train_score_logreg))
    print(roc_auc_score(y_small_test, y_test_score_logreg))

    ###################################################################################################################

    # 6. Постройте предсказания вероятностей победы команды Radiant для тестовой выборки с помощью лучшей из изученных
    # моделей (лучшей с точки зрения AUC-ROC на кросс-валидации). Убедитесь, что предсказанные вероятности адекватные
    # — находятся на отрезке [0, 1], не совпадают между собой (т.е. что модель не получилась константной).

    # Считайте таблицу с признаками из файла features.csv.
    test_data = pd.read_csv('data/features_test.csv', index_col='match_id')
    X_test = test_data.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                                    'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1)

    # Замените пропуски на нули с помощью функции fillna().
    X_test.fillna(value=0, inplace=True)

    scaler = StandardScaler()
    scaler.fit(X_test)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))

    # соберем список героев для "bag of words"
    heroes = test_data[['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                                    'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']].stack().unique()


    # Воспользуемся подходом "мешок слов" для кодирования информации о героях. Пусть всего в игре имеет N различных
    # героев. Сформируем N признаков, при этом i-й будет равен
    #   нулю, если i-й герой не участвовал в матче;
    #   единице, если i-й герой играл за команду Radiant;
    #   минус единице, если i-й герой играл за команду Dire.

    # N — количество различных героев в выборке
    N = heroes.max()
    X_pick = numpy.zeros((X_test.shape[0], N))

    for i, match_id in enumerate(X_test.index):
        for p in range(5):
            X_pick[i, test_data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, test_data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

    #print(X_pick)

    X_pick_df = pd.DataFrame(X_pick)
    X_pick_df.columns = range(1, N + 1)
    cols = [col for col in X_pick_df.columns if col in heroes]
    X_pick_df = X_pick_df[cols]

    X_fin = pd.concat([X_test_scaled, X_pick_df], axis=1)

    #print(X_fin.head())

    logistic = clf_logreg_grid.best_estimator_.predict_proba(X_fin)
    #print(logistic[:, 1])
    print('Min probability: %s' % str(min(logistic[:, 1])))
    print('Max probability: %s' % str(max(logistic[:, 1])))

if __name__ == "__main__":
    boosting()
    logreg()
