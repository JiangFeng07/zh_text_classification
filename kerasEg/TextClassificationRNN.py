#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from kerasEg.DataUtils import load_data, words_to_ids, text_to_sequence

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data_path', '/tmp/text_train.csv', 'train data path')

X_train, X_test, y_train, y_test = load_data(FLAGS.data_path)

## 建立 word—>id 字典
word_index = words_to_ids(X_train)

# 将每个词用词典中的数值代替
X_train_ids = np.array([text_to_sequence(ele, word_index) for ele in X_train])
X_test_ids = np.array([text_to_sequence(ele, word_index) for ele in X_test])

# 统一字符长度
X_train_data = tf.keras.preprocessing.sequence.pad_sequences(X_train_ids, value=word_index['<pad>'],
                                                             padding='post', maxlen=1000)
X_test_data = tf.keras.preprocessing.sequence.pad_sequences(X_test_ids, value=word_index['<pad>'],
                                                            padding='post', maxlen=1000)

## 类别用 one-hot 编码表示
label_index = dict()
for ele in set(y_train):
    label_index[ele] = len(label_index)
y_labels = list(label_index.values())
y_train_data = tf.keras.utils.to_categorical([label_index[ele] for ele in y_train], 2)
y_test_data = tf.keras.utils.to_categorical([label_index[ele] for ele in y_test], 2)

## model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(len(word_index) - 1, 16))
model.add(tf.keras.layers.LSTM(50, return_sequences=True))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

x_val = X_train_data[:300]
partial_x_train = X_train_data[300:]

y_val = y_train_data[:300]
partial_y_train = y_train_data[300:]

callbacks = [tf.keras.callbacks.TensorBoard(log_dir='/tmp/logs')]
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=100,
                    callbacks=callbacks,
                    validation_data=(x_val, y_val),
                    verbose=1)

model.save('my_model.h5')

results = model.evaluate(X_test_data, y_test_data)
print(results)

# history_dict = history.history
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
# loss = history_dict['loss']
# val_loss = history_dict['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
