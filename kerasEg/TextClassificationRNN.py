#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from kerasEg.DataUtils import words_to_ids, text_to_sequence

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('train_path', '/tmp/train.csv', 'train data path')
tf.flags.DEFINE_string('model_path', './model/my_model2.h5', 'model path')
tf.flags.DEFINE_integer('embedding_size', 16, 'embedding size')
tf.flags.DEFINE_integer('units', 50, 'lstm units')
tf.flags.DEFINE_integer('num_classes', 2, 'class number')


def create_rnn_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(len(word_index), FLAGS.embedding_size))
    model.add(tf.keras.layers.LSTM(FLAGS.units, return_sequences=True))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(FLAGS.num_classes, activation=tf.nn.softmax))
    return model


if __name__ == '__main__':

    data = pd.read_csv(FLAGS.train_path, header=None, sep='\t', error_bad_lines=False)
    texts = data.values[:, 1]
    labels = data.values[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, random_state=0)

    ## 建立 word—>id 字典
    word_index = words_to_ids(X_train)

    # 将每个词用词典中的数值代替
    X_train_ids = np.array([text_to_sequence(ele, word_index) for ele in X_train])
    X_test_ids = np.array([text_to_sequence(ele, word_index) for ele in X_test])

    # 统一字符长度
    X_train_data = tf.keras.preprocessing.sequence.pad_sequences(X_train_ids, value=word_index['<pad>'],
                                                                 padding='post', maxlen=500)
    X_test_data = tf.keras.preprocessing.sequence.pad_sequences(X_test_ids, value=word_index['<pad>'],
                                                                padding='post', maxlen=500)

    ## 类别用 one-hot 编码表示
    label_index = dict()
    for ele in set(y_train):
        label_index[ele] = len(label_index)
    y_train_data = tf.keras.utils.to_categorical([label_index[ele] for ele in y_train], 2)
    y_test_data = tf.keras.utils.to_categorical([label_index[ele] for ele in y_test], 2)

    # model
    model = create_rnn_model()

    ##自定义model
    # model = MyModel(num_classes=2, vocab_size=len(word_index) - 2, units=50)

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

    x_val = X_train_data[:3000]
    partial_x_train = X_train_data[3000:]

    y_val = y_train_data[:3000]
    partial_y_train = y_train_data[3000:]

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir='/tmp/logs')]
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=200,
                        callbacks=callbacks,
                        validation_data=(x_val, y_val),
                        verbose=1)

    model.save(FLAGS.model_path)
