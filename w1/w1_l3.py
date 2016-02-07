import pandas
import numpy

data = pandas.read_csv('data/titanic.csv', index_col='PassengerId')

index_col='PassengerId'

#print(data[:10])
#print(data.head())
#print(data['Pclass'])
#print(data['Pclass'].value_counts())

total_pass_count = data['Sex'].count()
print('Total pass count %s' % total_pass_count)

# Task 1
# Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.
print('Task 1')
print(len(data[data['Sex'] == 'male'].values))
print(len(data[data['Sex'] == 'female'].values))
print()

# Task 2
# Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров. Ответ приведите в процентах (число в
# интервале от 0 до 100, знак процента не нужен).
print('Task 2')
print(len(data[data['Survived'] == 1].values) * 100 / total_pass_count)
print(len(data[data['Survived'] == 0].values) * 100 / total_pass_count)
print()

# Task 3
# Какую долю пассажиры первого класса составляли среди всех пассажиров? Ответ приведите в процентах (число в интервале
# от 0 до 100, знак процента не нужен).
print('Task 3')
print(len(data[data['Pclass'] == 1].values) * 100 / total_pass_count)
print(len(data[data['Pclass'] != 1].values) * 100 / total_pass_count)
print()

# Task 4
# Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров. В качестве ответа приведите два
# числа через пробел.
print('Task 4')
print(data['Age'].sum() / len(data[data['Age'] > 0]))
print(data['Age'].median())
print()

# Task 5
# Коррелируют ли число братьев/сестер с числом родителей/детей? Посчитайте корреляцию Пирсона между признаками
# SibSp и Parch.
print('Task 5')
print(numpy.corrcoef(data['SibSp'], data['Parch'])[0, 1])
print()

# Task 6
# Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name) его личное имя
# (First Name). Попробуйте вручную разобрать несколько
# значений столбца Name и выработать правило для извлечения имен, а также разделения их на женские и мужские.
print('Task 6')
females = data[data['Sex'] == 'female']
female_names = females['Name']
print(female_names)
females['FirstName'] = females.Name.apply(lambda x: x.split(',')[0])
females['FirstName1'] = females.Name.apply(lambda x: x.split(',')[0])
#print(females.groupby('FirstName').count())
#print(females['FirstName'])
#print(females['FirstName'].describe())
print()