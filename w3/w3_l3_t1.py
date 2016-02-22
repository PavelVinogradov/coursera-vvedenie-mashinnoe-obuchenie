import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve

import numpy as np

def read_classification():
    input_data = []

    with open('data/l3_classification.csv') as csvfile:
        reader = csv.reader(csvfile)
        is_header = True

        for row in reader:
            if is_header:
                is_header = False
            else:
                input_data.append(row)

    return input_data


def read_scores():
    input_data = []

    with open('data/l3_scores.csv') as csvfile:
        reader = csv.reader(csvfile)
        is_header = True

        for row in reader:
            if is_header:
                is_header = False
            else:
                input_data.append(row)

    return input_data


def matrix(predictions):
    result = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}

    for row in predictions:
        if row[0] == '1' and row[1] == '1':
            result['TP'] += 1
        if row[0] == '1' and row[1] == '0':
            result['FN'] += 1
        if row[0] == '0' and row[1] == '1':
            result['FP'] += 1
        if row[0] == '0' and row[1] == '0':
            result['TN'] += 1

    return result


def convert(predictions):
    pred = {'y': [], 'x': []}
    for row in predictions:
        pred['y'].append(int(row[0]))
        pred['x'].append(int(row[1]))

    return pred


def score(predictions):
    accuracy = accuracy_score(predictions['y'], predictions['x'])
    precision = precision_score(predictions['y'], predictions['x'])
    recall = recall_score(predictions['y'], predictions['x'])
    f1 = f1_score(predictions['y'], predictions['x'])

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def roc_score(predictions):
    logreg = roc_auc_score([int(y) for y in predictions[:, 0]], [float(w) for w in predictions[:, 1]])
    svm = roc_auc_score([int(y) for y in predictions[:, 0]], [float(w) for w in predictions[:, 2]])
    knn = roc_auc_score([int(y) for y in predictions[:, 0]], [float(w) for w in predictions[:, 3]])
    tree = roc_auc_score([int(y) for y in predictions[:, 0]], [float(w) for w in predictions[:, 4]])

    return {'logreg': logreg, 'svm': svm, 'knn': knn, 'tree': tree}


def precision_recall(predictions):
    prc_logreg = precision_recall_curve([int(y) for y in predictions[:, 0]], [float(w) for w in predictions[:, 1]])
    prc_svm = precision_recall_curve([int(y) for y in predictions[:, 0]], [float(w) for w in predictions[:, 2]])
    prc_knn = precision_recall_curve([int(y) for y in predictions[:, 0]], [float(w) for w in predictions[:, 3]])
    prc_tree = precision_recall_curve([int(y) for y in predictions[:, 0]], [float(w) for w in predictions[:, 4]])

    print('logreg => %s' % round(max(prc_logreg[0][prc_logreg[1] >= 0.7]), 2))
    print('svm => %s' % round(max(prc_svm[0][prc_svm[1] >= 0.7]), 2))
    print('knn => %s' % round(max(prc_knn[0][prc_knn[1] >= 0.7]), 2))
    print('tree => %s' % round(max(prc_tree[0][prc_tree[1] >= 0.7]), 2))


def precision_recall_process(name, prc):
    #for (prec, recall, thresh) in [zip(prc)]:
    #for row in [row for row in prc if row[1] > 0.7]:
    for row in prc:
        print('\t%s' % (row))
        #print('\t%s -> %s -> %s' % (prec, recall, thresh))

    #for (precision, recall) in [zip(row[0:2]) for row in prc]:# if row[1] > 0.7]:
    #    print('\tprec = %s, recall = %s' % (precision, recall))

def part_one():
    data = read_classification()
    print(data)
    print("===============================")
    print("Task One:\n\t %s" % matrix(data))
    print("===============================")
    print("Task Two:\n\t %s" % score(convert(data)))


def part_two():
    data = np.array(read_scores())

    print(roc_score(data))
    precision_recall(data)

part_two()
