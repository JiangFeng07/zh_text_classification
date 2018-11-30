#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import argparse
import collections

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='biyu_model/train2.csv')
parser.add_argument('--test_size', type=float, default='0.1')
FLAGS, unparser = parser.parse_known_args()



# 加载数据并随机打乱
def load_data(file_name, sep=' ', sep1=',', isCharacter=False):
    label_list = []
    features_list = []
    with tf.gfile.GFile(file_name, 'r') as f:
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
def build_word_dic(words_list, label_list, vocab_size=5000):
    word_dic = dict()
    word_dic['pad'] = 0
    word_dic['unk'] = 1
    all_words = []
    for words in words_list:
        all_words.extend(words)
    counter = collections.Counter(all_words).most_common(vocab_size - 2)
    words, _ = list(zip(*counter))
    for word in words:
        word_dic[word] = len(word_dic)
    label_set = set(label_list)
    label_dic = dict()
    for label in label_set:
        label_dic[label] = len(label_dic)
    return words, word_dic, label_set, label_dic
