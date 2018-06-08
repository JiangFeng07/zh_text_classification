#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import argparse
import collections
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='/tmp/1.csv')
FLAGS, unparser = parser.parse_known_args()


# 加载数据并随机打乱
def load_data(sep=' ', sep1=',', isCharacter=False):
    label_list = []
    features_list = []
    with tf.gfile.GFile(FLAGS.train_file, 'r') as f:
        for line in f.readlines():
            fields = line.strip().split(sep)
            if len(fields) != 2:
                continue
            label = fields[0]
            features = fields[1]
            label_list.append(label)
            if isCharacter:
                features_list.append(list(features))
            else:
                features_list.append(features.split(sep1))
    indices = np.random.permutation(np.arange(len(features_list)))
    label_list = np.array(label_list)[indices]
    features_list = np.array(features_list)[indices]
    return label_list, features_list


# 词汇->id 映射
def build_word_dic(words_list, label_list, vocab_size=50):
    word_dic = dict()
    word_dic['pad'] = 0
    word_dic['unk'] = 1
    all_words = []
    for words in words_list:
        all_words.extend(words)
    counter = collections.Counter(all_words).most_common(vocab_size)
    words, _ = list(zip(*counter))
    for word in words:
        word_dic[word] = len(word_dic)
    label_set = set(label_list)
    label_dic = dict()
    for label in label_set:
        label_dic[label] = len(label_dic)
    return words, word_dic, label_set, label_dic


def build_dic_hash_table(word_dic, label_dic):
    word_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(list(word_dic.keys()), list(word_dic.values())), word_dic['unk'])
    label_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(list(label_dic.keys()), list(label_dic.values())), -1)
    return word_table, label_table


def gen():
    with tf.gfile.GFile(FLAGS.train_file, 'r') as f:
        lines = [line.strip().split(' ') for line in f]
        index = 0
        while True:
            label = lines[index][0]
            features = lines[index][1].split(',')
            yield (label, features)
            index += 1
            if index == len(lines):
                index = 0


def train_input_fn(shuffle_size, batch_size):
    dataset = tf.data.Dataset.from_generator(gen, (tf.string, tf.string), (tf.TensorShape([]), tf.TensorShape([None])))
    return dataset.shuffle(shuffle_size).repeat().padded_batch(batch_size=batch_size, padded_shapes=(
        tf.TensorShape([]), tf.TensorShape([None])), padding_values=('', 'pad'))


# def train_input_fn(label_list, features_list, shuffle_size, batch_size):
#     dataset = tf.data.Dataset.from_tensor_slices((label_list, features_list))
#     dataset = dataset.shuffle(shuffle_size).repeat().batch(batch_size)
#     return dataset


# def build_table_from_text_file(filepath):
#     return tf.contrib.lookup.HashTable(
#         tf.contrib.lookup.TextFileInitializer(filepath, tf.string, 0, tf.int64, 1, delimiter=" "), -1)


def test_dataset_from_generator():
    label_list, words_list = load_data()
    _, word_dic, _, label_dic = build_word_dic(words_list, label_list)
    word_table, label_table = build_dic_hash_table(word_dic, label_dic)

    dataset = train_input_fn(16, 2)
    dataset = dataset.map(lambda x, y: (label_table.lookup(x), word_table.lookup(y)))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.tables_initializer())
        for i in range(6):
            try:
                print(sess.run(next_element))
                # print(sess.run(label_table.lookup(next_element[0])))
                # print(sess.run(word_table.lookup(next_element[1])))
            except tf.errors.OutOfRangeError:
                break


def test_hash_table():
    label_list, words_list = load_data()
    _, word_dic, _, label_dic = build_word_dic(words_list, label_list)
    word_table, label_table = build_dic_hash_table(word_dic, label_dic)
    word_out = word_table.lookup(tf.constant(list(word_dic.keys())))
    label_out = label_table.lookup(tf.constant(list(label_dic.keys())))
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        print(sess.run(word_out))
        print(sess.run(label_out))


if __name__ == '__main__':
    data = load_data()
    print(data)
    # test_dataset_from_generator()
    # test_hash_table()
