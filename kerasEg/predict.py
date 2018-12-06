#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from kerasEg.DataUtils import words_to_ids, text_to_sequence

# tf.enable_eager_execution()
if __name__ == '__main__':
    data = pd.read_csv("/tmp/train.csv", header=None, sep='\t', error_bad_lines=False)
    texts = data.values[:, 1]
    labels = data.values[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, random_state=0)

    word_index = words_to_ids(X_train)
    # print(word_index)

    data = pd.read_csv('/tmp/content_valid.csv', header=None, sep='\t', error_bad_lines=False)
    texts = data.values[:, 2]
    labels = data.values[:, 0]
    sentiment = data.values[:, 1]

    # 将每个词用词典中的数值代替
    X_train_ids = np.array([text_to_sequence(ele, word_index) for ele in texts])

    # 统一字符长度
    X_train_data = tf.keras.preprocessing.sequence.pad_sequences(X_train_ids, value=word_index['<pad>'],
                                                                 padding='post', maxlen=500)
    new_model = tf.keras.models.load_model('my_model.h5')
    result = new_model.predict(X_train_data[:500], batch_size=100)
    sess = tf.Session()
    result = sess.run(tf.argmax(result, 1))
    pd = pd.DataFrame({'A': labels[:500], 'B': result, 'C': sentiment[:500]})
    pd.to_csv("/tmp/1.csv", sep='\t', index=False)
