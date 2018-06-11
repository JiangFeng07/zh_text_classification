#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import argparse

from data_utils import load_data, build_word_dic
from sklearn.model_selection import train_test_split
import tensorflow as tf

from rnn_model2 import TextRNN

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='biyu_model/train2.csv')
parser.add_argument('--test_size', type=float, default='0.1')
parser.add_argument('--embedding_size', type=int, default=200)
parser.add_argument('--hidden_layers', type=int, default=2)
parser.add_argument('--hidden_units', type=int, default=256)
parser.add_argument('--number_classes', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--sequence_length', type=int, default=300)
parser.add_argument('--vocab_size', type=int, default=5000)
FLAGS, unparser = parser.parse_known_args()


def build_dic_hash_table(word_dic, label_dic):
    word_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(list(word_dic.keys()), list(word_dic.values())), word_dic['unk'])
    label_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(list(label_dic.keys()), list(label_dic.values())), -1)
    return word_table, label_table


def gen(x, y):
    index = 0
    while True:
        label = y[index]
        features = x[index]
        yield (label, features)
        index += 1
        if index == len(x):
            index = 0


def train_input_fn(shuffle_size, batch_size, x, y):
    dataset = tf.data.Dataset.from_generator(lambda: gen(x, y), (tf.string, tf.string),
                                             (tf.TensorShape([]), tf.TensorShape([None])))
    return dataset.shuffle(shuffle_size).repeat().padded_batch(batch_size=batch_size, padded_shapes=(
        tf.TensorShape([]), tf.TensorShape([None])), padding_values=('', 'pad'))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


if __name__ == "__main__":

    label_list, words_list = load_data(FLAGS.train_file)
    train_x, test_x, train_y, test_y = train_test_split(words_list, label_list, test_size=0.1, random_state=0)
    _, word_dic, _, label_dic = build_word_dic(train_x, train_y)
    word_table, label_table = build_dic_hash_table(word_dic, label_dic)
    dataset = train_input_fn(len(train_x), 128, train_x, train_y)
    dataset = dataset.map(lambda y, x: (label_table.lookup(y), word_table.lookup(x)))
    iterator = dataset.make_initializable_iterator()
    model = TextRNN(FLAGS.embedding_size, FLAGS.hidden_layers, FLAGS.hidden_units, FLAGS.number_classes,
                    FLAGS.learning_rate, FLAGS.vocab_size, iterator)
    # next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            try:
                sess.run(model.optimizer, feed_dict={model.keep_prob: 0.8})
                if i % 10 == 0:
                    print(sess.run([model.accuracy, model.loss], feed_dict={model.keep_prob: 1}))
            except tf.errors.OutOfRangeError:
                break
