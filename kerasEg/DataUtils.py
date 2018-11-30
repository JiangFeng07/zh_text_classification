#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import logging
import os
import time
import pandas as pd
import re
from sklearn.model_selection import train_test_split


def data_process(file_path, tag):
    files = os.listdir(file_path)
    with open('/tmp/%s.csv' % tag, 'w', encoding='utf-8') as out:
        for file in files:
            with open(os.path.join(file_path, file), 'r', encoding='utf-8') as f:
                try:

                    out.write("%s\t%s\n" % (tag, extract_word(f.read(), '[^\u4e00-\u9fa5A-Za-z]+')))
                except:
                    continue


def load_data(file_path):
    data = pd.read_csv(file_path, header=None, sep='\t', error_bad_lines=False)
    texts = data.values[:, 1]
    lables = data.values[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(texts, lables, test_size=0.1, random_state=0)
    return X_train, X_test, y_train, y_test


def extract_word(text, pattern):
    zh_pattern = re.compile(pattern)
    return ''.join(zh_pattern.split(text))


if __name__ == '__main__':
    start_time = time.time()
    data_process('/Users/lionel/Desktop/data/nlp/THUCNews/THUCNews/体育', 'sport')
    data_process('/Users/lionel/Desktop/data/nlp/THUCNews/THUCNews/科技', 'tech')
    end_time = time.time()
    print('process data take %d seconds' % (end_time - start_time))
    # file_path = '/tmp/text_train.csv'
    #
    # X_train, X_test, y_train, y_test = load_data(file_path)
    # print(len(X_train), len(y_train))
    # print(len(X_test), len(y_test))
