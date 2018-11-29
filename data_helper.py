#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: lionel

import collections
import tensorflow.contrib.keras as kr
import numpy as np
import tensorflow as tf


def read_data(file_name, sep=' ', sep1=',', chinese_only=False):
    texts = []
    contents = []
    labels = []
    with tf.gfile.GFile(file_name, 'r') as file:
        for line in file.readlines():
            fields = line.decode('utf-8').strip().split(sep)
            if len(fields) != 2:
                continue
            labels.append(fields[0])
            texts.append(fields[1])
            if chinese_only is True:
                # contents.append(list(extract_chinese_word(fields[1])))
                contents.append(list(fields[1]))
            else:
                contents.append(fields[1].split(sep1))
    return contents, labels, texts


def word_to_id(contents, labels, vocab_size=5000):
    all_data = []
    for content in contents:
        all_data.extend(content)
    counter = collections.Counter(all_data).most_common(vocab_size - 2)
    words, _ = list(zip(*counter))
    words2 = []
    for word in words:
        words2.append(word)
    words2.sort()
    word2id = dict()
    word2id['pad'] = 0
    word2id['unk'] = 1
    for word in words2:
        word2id[word] = len(word2id)
    labels2 = []
    for ele in labels:
        if ele not in labels2:
            labels2.append(ele)
    label2id = dict()
    for ele in labels2:
        label2id[ele] = len(label2id)
    return words2, word2id, labels2, label2id


def save_words(word2id, wordfile):
    return save_text(word2id, wordfile)


def save_labels(label2id, lablefile):
    return save_text(label2id, lablefile)


def save_text(dic, file):
    with tf.gfile.GFile(file, 'w') as f:
        for ele in dic:
            f.write(ele + ':' + str(dic[ele]) + '\n')


def words_to_dic(wordfile):
    return text_to_dic(wordfile)


def label_to_dic(labelfile):
    return text_to_dic(labelfile)


def text_to_dic(file):
    dic = dict()
    with tf.gfile.GFile(file, 'r') as f:
        for line in f.readlines():
            fields = line.strip().decode('utf-8').split(':')
            if len(fields) != 2:
                continue
            dic[fields[0]] = int(fields[1])
    return dic


# def extract_chinese_word(text):
    # line = text.strip().decode('utf-8', 'ignore')
    # zh_pattern = re.compile(ur'[^\u4e00-\u9fa5]+')
    # return ''.join(zh_pattern.split(text))


def process_file(contents, labels, word2id, category2id, max_length=600):
    data_id, labels_id = [], []
    for i in range(len(contents)):
        tmp = []
        for x in contents[i]:
            if x in word2id:
                tmp.append(word2id[x])
            else:
                tmp.append(word2id['unk'])
        data_id.append(tmp)
        labels_id.append([category2id[labels[i]]])
    x_padding = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', value=word2id['pad'])
    y_padding = kr.utils.to_categorical(labels_id)

    return x_padding, y_padding


def process_predict_file(pred_contents, word2id, max_length=600):
    data_id = []
    for i in range(len(pred_contents)):
        tmp = []
        for x in pred_contents[i]:
            if x in word2id:
                tmp.append(word2id[x])
            else:
                tmp.append(word2id['unk'])
        data_id.append(tmp)
    x_padding = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', value=word2id['pad'])
    return x_padding


def generate_batch(x, y, batch_size=128):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size + 1)

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


if __name__ == '__main__':
    contents, labels, texts = read_data("/tmp/1.csv", chinese_only=True)

    # print contents
