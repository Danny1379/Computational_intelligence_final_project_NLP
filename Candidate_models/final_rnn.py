import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

vocab_size = 40000
embedding_dim = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"


def load_data():
    path = "train.csv"
    df = pd.read_csv(path)
    print(np.shape(df))
    return df


def get_labels_features(df):
    df.Text = df.Text.str.replace('\d', '')
    df.Text = df.Text.str.replace('\n', ' ')
    df.Text = df.Text.str.replace('.', " ")
    df.Text = df.Text.str.replace(',', " ")
    x = df['Text']
    y = df['Category']
    print(np.shape(x), np.shape(y))
    return x, y


def preprocess_encode(y):
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y)
    y = np_utils.to_categorical(y)
    return y


def split(x, y):
    return train_test_split(x, y, test_size=0.10)


def tokenize(x_train, x_test):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(x_train)
    training_sequences = tokenizer.texts_to_sequences(x_train)
    max_length = get_sequence_length(training_sequences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    testing_sequences = tokenizer.texts_to_sequences(x_test)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return training_padded, testing_padded, max_length


# find average sequence length might go for maximum length !

def get_sequence_length(training_sequences):
    sum = 0
    for i in range(len(training_sequences)):
        sum += len(training_sequences[i])
    max_length = int(sum / len(training_sequences))
    print("sequence length is : ", max_length)
    return max_length


def get_array(training_padded, y_train, testing_padded, y_test):
    training_padded = np.asarray(training_padded)
    training_labels = np.asarray(y_train)
    testing_padded = np.asarray(testing_padded)
    testing_labels = np.asarray(y_test)
    return training_padded, training_labels, testing_padded, testing_labels


def get_model(max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.SpatialDropout1D(0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_length)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)),
        tf.keras.layers.Dense(34, activation="softmax")
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, training_padded, training_labels, testing_padded, testing_labels):
    num_epochs = 10
    history = model.fit(training_padded, training_labels, epochs=num_epochs,
                        validation_data=(testing_padded, testing_labels), verbose=1, batch_size=256)


def main():
    df = load_data()
    x, y = get_labels_features(df)
    y = preprocess_encode(y)
    x_train, x_test, y_train, y_test = split(x, y)
    x_train, x_test, max_length = tokenize(x_train, x_test)
    x_train, y_train, x_test, y_test = get_array(x_train, y_train, x_test, y_test)
    model = get_model(max_length)
    train_model(model, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
