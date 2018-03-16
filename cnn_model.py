#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel

import tensorflow as tf


class TextCNN(object):
    def __init__(self, embedding_size, number_classes, sequence_length, learning_rate, filter_sizes, num_filters,
                 vocab_size, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, number_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # embedding层
        with tf.name_scope('embedding'), tf.device('/cpu:0'):
            embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='embedding')
            self.embedded = tf.nn.embedding_lookup(embedding, self.input_x)
            self.embedded_expanded = tf.expand_dims(self.embedded, -1)
        pooled_outputs = []

        # conv and max_pool层
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv_maxpool-%s' % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')

                conv = tf.nn.conv2d(self.embedded_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name='pool')
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

        # fully connected layer and softmax
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[num_filters_total, number_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[number_classes], name='b'))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name='score')
            self.prediction = tf.nn.softmax(self.logits)

        with tf.name_scope('optimizer'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy) + l2_reg_lambda * l2_loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.input_y, 1),
                                          name='correct_prediction')
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
