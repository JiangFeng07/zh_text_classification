#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel

import tensorflow as tf


class RNNModel(tf.keras.Model):
    def __init__(self, num_classes=2, vocab_size=1000, embedding_size=16, units=32):
        super(RNNModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size + 1, embedding_size)
        self.LSTM_layer = tf.keras.layers.LSTM(units=units, return_sequences=True)
        self.GlobalAveragePooling_layer = tf.keras.layers.GlobalAveragePooling1D()
        self.dense_layer = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.LSTM_layer(x)
        x = self.GlobalAveragePooling_layer(x)
        return self.dense_layer(x)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
