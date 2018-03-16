#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import tensorflow as tf


class TextRNN(object):
    def __init__(self, embedding_size, hidden_layers, hidden_units, number_classes, learning_rate, sequence_length,
                 vocab_size):
        self.hidden_units = hidden_units
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, number_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], -1.0, 1.0), name='embedding')
            embed = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('multi_rnn'):
            cells = [self.drop_wrapper() for _ in range(hidden_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embed, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, hidden_units, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, number_classes, name='fc2')
            self.predict = tf.nn.softmax(self.logits)  # 预测类别

        with tf.name_scope("optimizer"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), tf.argmax(self.predict, 1))
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
