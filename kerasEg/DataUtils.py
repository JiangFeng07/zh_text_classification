#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import re

from tensorflow.python.keras.preprocessing.text import Tokenizer


# def data_process(file_path, tag):
#     files = os.listdir(file_path)
#     with open('/tmp/%s.csv' % tag, 'w', encoding='utf-8') as out:
#         for file in files:
#             with open(os.path.join(file_path, file), 'r', encoding='utf-8') as f:
#                 try:
#                     out.write("%s\t%s\n" % (tag, extract_word(f.read(), '[^\u4e00-\u9fa5A-Za-z]+')))
#                 except:
#                     continue


def data_process2(file_path, out_path):
    with open(out_path, 'w', encoding='utf-8') as out:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.split('\t')
                if len(fields) != 4:
                    continue
                words = extract_word(fields[3].strip(), '[^\u4e00-\u9fa5A-Za-z]+')
                if len(words) <= 0:
                    continue
                out.write("%s\t%s\t%s" % (fields[0], words, fields[3]))


def words_to_ids(train_data):
    """
    建立 word->id 字典
    :param train_data: 文本文件
    :return: word->id 字典
    """
    tokenizer = Tokenizer(lower=True, char_level=True)
    tokenizer.fit_on_texts(train_data)
    word_index = tokenizer.word_index
    word_index = {k: (v + 1) for k, v in word_index.items()}
    word_index['<pad>'] = 0
    word_index['<unk>'] = 1
    return word_index


def text_to_sequence(text, word_index):
    sequence = []
    for word in list(text):
        sequence.append(word_index.get(word, word_index.get('<unk>')))
    return sequence


def extract_word(text, pattern):
    """
    :param text: 文本
    :param pattern: 正则表达式
    :return: 返回取满足正则的文本
    """
    zh_pattern = re.compile(pattern)
    return ''.join(zh_pattern.split(text))
