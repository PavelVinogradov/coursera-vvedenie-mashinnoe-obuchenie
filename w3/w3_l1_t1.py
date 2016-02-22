import csv
from sklearn.svm import SVC

def read():
    X = []
    y = []
    with open('data/l1_svm-data.csv') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            y.append(row[0])
            X.append(row[1:])

    return (X, y)


(X, y) = read()

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X, y)
print(clf.support_)