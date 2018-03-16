#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import tensorflow as tf


class BiLstmModel(object):
    def __init__(self, embedding_size, hidden_layers, hidden_units, number_classes, learning_rate, sequence_length,
                 vocab_size):
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, number_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope('fw_rnn'):
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(hidden_units) for _ in range(hidden_layers)]
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                           output_keep_prob=self.keep_prob)

        with tf.name_scope('bw_rnn'):
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(hidden_units) for _ in range(hidden_layers)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list),
                                                           output_keep_prob=self.keep_prob)

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.get_variable('embedding', [vocab_size, embedding_size])
            embed = tf.nn.embedding_lookup(embedding, self.input_x)

        inputs = tf.transpose(embed, [1, 0, 2])

        inputs = tf.reshape(inputs, [-1, hidden_units])

        inputs = tf.split(inputs, sequence_length, 0)

        with tf.name_scope('bi_rnn'):
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, inputs,
                                                                    dtype=tf.float32)

        with tf.name_scope('score'):
            fc_w = tf.Variable(tf.truncated_normal([2 * hidden_units, number_classes], stddev=0.1), name='fc_w')
            fc_b = tf.Variable(tf.zeros([number_classes], name='fc_b'))
            self.logits = tf.matmul(outputs[-1], fc_w) + fc_b
            self.predict = tf.nn.softmax(self.logits)

        with tf.name_scope('optimizer'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
