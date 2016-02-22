import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

train_data_path = 'data/salary-train.csv'
#train_data_path = 'data/salary-train-small.csv'
test_data_path = 'data/salary-test-mini.csv'

transformer = TfidfVectorizer(min_df=5)
dict_encoder = DictVectorizer()


def read_data_as_df(data_path):
    return pd.read_csv(data_path)


def train_description_text_vectorizer(data):
    return transformer.fit_transform(data['FullDescription'])


def test_description_text_vectorizer(data):
    return transformer.transform(data['FullDescription'])


def train_dict_values_vectorizer(data):
    X_train_categ = dict_encoder.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
    return X_train_categ


def test_dict_values_vectorizer(data):
    X_test_categ = dict_encoder.transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
    return X_test_categ


def fill_na(data):

    data['LocationNormalized'].fillna('nan', inplace=True)
    data['ContractTime'].fillna('nan', inplace=True)

    return data


def normalize_train_data(data):
    data['FullDescription'] = data['FullDescription'].apply(str.lower)
    data['LocationNormalized'] = data['LocationNormalized'].apply(str.lower)
    data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    data = fill_na(data)

    X_train_corpus = train_description_text_vectorizer(data)
    X_train_categ = train_dict_values_vectorizer(data)

    X_train = hstack((X_train_corpus, X_train_categ))

    return X_train

def normalize_test_data(data):
    data['FullDescription'] = data['FullDescription'].apply(str.lower)
    data['LocationNormalized'] = data['LocationNormalized'].apply(str.lower)
    data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    data = fill_na(data)

    X_test_vocab = test_description_text_vectorizer(data)
    X_test_categ = test_dict_values_vectorizer(data)
    X_test = hstack((X_test_vocab, X_test_categ))

    return X_test


def train_ridge_model(X, y):
    model = Ridge(alpha=1)
    clf = model.fit(X, y)

    return clf

if __name__ == "__main__":
    train_data_raw = read_data_as_df(train_data_path)
    print("Normalize train data")
    train_data_norm = normalize_train_data(train_data_raw)
    print("Fit Ridge model")
    clf = train_ridge_model(train_data_norm, train_data_raw['SalaryNormalized'])

    test_data_raw = read_data_as_df(test_data_path)
    print("Normalize test data")
    test_data_norm = normalize_test_data(test_data_raw)

    print("Predict")
    predictions = clf.predict(test_data_norm)
    print(predictions)
