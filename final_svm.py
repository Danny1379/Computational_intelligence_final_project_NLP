from __future__ import unicode_literals

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2
from hazm import *


def load_data():
    path = "train.csv"
    df = pd.read_csv(path)
    print(np.shape(df))
    return df


def get_labels_features(df):
    x = df['Text']
    y = df['Category']
    print(np.shape(x), np.shape(y))
    return x, y


def preprocess_encode(y):
    label_enc = LabelEncoder()
    y_labeled = label_enc.fit_transform(y)
    return y_labeled


def split(x, y):
    return train_test_split(x, y, test_size=0.10)


# this will take a while i suggest reading a book
def tf_idf(x_train, x_test):
    normalizer = Normalizer()
    #print(x_train.shape)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode",
                                 preprocessor=normalizer.normalize, tokenizer=word_tokenize)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    #print(x_train.shape)
    return x_train, x_test


# reduce number of features to improve accuracy and reduce memory usage
def get_best_features(x_train, y_train, x_test):
    select_best = SelectKBest(chi2, k=400000)
    select_best.fit(x_train, y_train)
    x_train = select_best.transform(x_train)
    x_test = select_best.transform(x_test)
    return x_train, x_test


# this will take quite a while i suggest getting a cup of coffee
def svm(x_train, y_train):
    linear_svm = LinearSVC()
    linear_svm.fit(x_train, y_train)
    return linear_svm


def get_accuracy(linear_svm, x_test, y_test):
    predictions = linear_svm.predict(x_test)
    accuracy = np.mean(predictions == y_test)
    return accuracy


def main():
    df = load_data()
    x, y = get_labels_features(df)
    y = preprocess_encode(y)
    x_train, x_test, y_train, y_test = split(x, y)
    x_train, x_test = tf_idf(x_train, x_test)
    x_train, x_test = get_best_features(x_train, y_train, x_test)
    model = svm(x_train, y_train)
    accuracy = get_accuracy(model, x_test, y_test)
    print("our model accuracy is :", accuracy)


if __name__ == '__main__':
    main()
