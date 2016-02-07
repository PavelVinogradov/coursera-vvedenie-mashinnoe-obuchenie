import pandas
import numpy
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess


index_col = 'PassengerId'
features = ['Pclass', 'Fare', 'Age', 'Sex']

data = pandas.read_csv('data/titanic.csv', index_col=index_col)

def data_prepare(df):
    df_mod = df.copy()

    df_mod = encode_sex(df_mod)
    df_mod = remove_nan_age(df_mod)

    return df_mod


# Replace strings in 'sex' field with numbers
def encode_sex(df):
    df_mod = df.copy()
    sexs = df_mod['Sex'].unique()
    map_to_int = {name: n for n, name in enumerate(sexs)}
    df_mod['Sex'] = df_mod['Sex'].replace(map_to_int)

    return df_mod


# Remove records with NaN age
def remove_nan_age(df):
    df_mod = df.copy()

    #df_mod = df_mod[not numpy.isnan(df_mod['Age'])]
    df_mod = df_mod[df_mod['Age'] > 0]

    return df_mod


# Fit the decision tree
def dtree_fit(df):
    # Pclass, Fare, Age, Sex
    x = df[features]
    y = df['Survived']

    dt = DecisionTreeClassifier(random_state=241)
    dt.fit(x, y)

    return dt


# Create tree visualization
def visualize_tree(tree):
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=features)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

df = data_prepare(data)
dt = dtree_fit(df)
#visualize_tree(dt)
importances = dt.feature_importances_
print(importances)