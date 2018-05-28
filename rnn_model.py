#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class TextRNN(object):
    def __init__(self, embedding_size, hidden_layers, hidden_units, number_classes, learning_rate, sequence_length,
                 vocab_size):
        self.hidden_units = hidden_units
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, number_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        length = tf.reduce_sum(tf.sign(self.input_x), axis=1)

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.get_variable(name='embedding', shape=[vocab_size, embedding_size], dtype=tf.float32,initializer=xavier_initializer())
            # embedding = tf.get_variable(name='embedding', shape=[vocab_size, embedding_size], dtype=tf.float32)
            embed = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('multi_rnn'):
            cells = [self.drop_wrapper() for _ in range(hidden_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, sequence_length=length, inputs=embed, dtype=tf.float32)
            batch_range = tf.range(tf.shape(self.input_x)[0])
            indices = tf.stack([batch_range, length - 1], axis=1)
            self.last = tf.gather_nd(_outputs, indices)

            # self.last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(self.last, hidden_units, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, number_classes, name='fc2')
            self.prediction = tf.nn.softmax(self.logits)  # 预测类别

        with tf.name_scope("optimizer"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), tf.argmax(self.prediction, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def lstm(self):
        with tf.name_scope('lstm_cell'):
            # lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_units, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.GRUCell(self.hidden_units)
        return lstm_cell

    def drop_wrapper(self):
        lstm_cell = self.lstm()
        with tf.name_scope('drop_wrapper'):
            drop_wrapper = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        return drop_wrapper
