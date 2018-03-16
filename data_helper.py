#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: lionel

import collections
import tensorflow.contrib.keras as kr
import numpy as np
import tensorflow as tf


def read_data(file_name):
    contents = []
    labels = []
    with tf.gfile.GFile(file_name, 'r') as file:
        for line in file.readlines():
            fields = line.strip().split("\t")
            labels.append(fields[0])
            contents.append(fields[1].split(","))
    return contents, labels


def word_to_id(contents, labels, vocab_size=5000):
    all_data = []
    for content in contents:
        all_data.extend(content)
    counter = collections.Counter(all_data).most_common(vocab_size)
    words, _ = list(zip(*counter))
    word2id = dict(zip(words, range(len(words))))
    labels2 = []
    for ele in labels:
        if ele not in labels2:
            labels2.append(ele)
    label2id = dict(zip(labels2, range(len(labels2))))
    return words, word2id, labels2, label2id


def process_file(file_name, word2id, category2id, max_length=600):
    contents, labels = read_data(file_name)

    data_id, labels_id = [], []

    for i in range(len(contents)):
        data_id.append([word2id[x] for x in contents[i] if x in word2id])
        labels_id.append([category2id[labels[i]]])
    x_padding = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_padding = kr.utils.to_categorical(labels_id)

    return x_padding, y_padding


def generate_batch(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size + 1)

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
