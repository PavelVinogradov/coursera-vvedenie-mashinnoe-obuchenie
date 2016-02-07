from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

import csv


# read input data from csv files
def read_data(type='train'):
    data_x = []
    data_y = []

    with open('data/perceptron-%s.csv' % type) as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            data_x.append([float(x) for x in row[1:]])
            data_y.append(float(row[0]))

    return (data_x, data_y)


def test():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    clf = Perceptron()
    clf.fit(X, y)

    predictions = clf.predict(X)

    print("Predictions: %s" % predictions)

    print("Accuracy: %s" % accuracy_score(y, predictions))


def solve(train_set_x, train_set_y, test_set_x, test_set_y):
    clf = Perceptron(random_state=241)
    clf.fit(X=train_set_x, y=train_set_y)
    prediction = clf.predict(test_set_x)

    accuracy = accuracy_score(test_set_y, prediction)

    return accuracy

scaler = StandardScaler()

train_set_x, train_set_y = read_data('train')
test_set_x, test_set_y = read_data('test')
train_set_x_scaled = scaler.fit_transform(train_set_x)
test_set_x_scaled = scaler.transform(test_set_x)

accuracy_raw = solve(train_set_x, train_set_y, test_set_x, test_set_y)
accuracy_scaled = solve(train_set_x_scaled, train_set_y, test_set_x_scaled, test_set_y)

print("Accuracy raw: %s" % accuracy_raw)
print("Accuracy scaled: %s" % accuracy_scaled)

print("Answer %s" % (accuracy_scaled-accuracy_raw))
