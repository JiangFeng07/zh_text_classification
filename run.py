#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import argparse
import os
import sys
import time
from datetime import timedelta

import tensorflow as tf
from sklearn import metrics

from cnn_model import TextCNN
from data_helper import *
from rnn_model import TextRNN

FLAGS = None


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = generate_batch(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(FLAGS.train_data, word_to_id, label_to_id, FLAGS.sequence_length)
    x_val, y_val = process_file(FLAGS.valid_data, word_to_id, label_to_id, FLAGS.sequence_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(2):
        print('Epoch:', epoch + 1)
        batch_train = generate_batch(x_train, y_train, batch_size=FLAGS.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, 0.8)

            if total_batch % 10 == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % 50 == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=FLAGS.save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optimizer, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(FLAGS.test_data, word_to_id, label_to_id, FLAGS.sequence_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=FLAGS.save_path)  # 读取保存的模型
    # saver = tf.train.import_meta_graph("model/model.ckpt/model.ckpt.meta")
    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 50
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(tf.argmax(model.prediction, 1), feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=labels))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_units', type=int, default=128)
    parser.add_argument('--number_classes', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--sequence_length', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=70)
    parser.add_argument('--vocab_size', type=int, default=5000)
    parser.add_argument('--filter_sizes', type=list, default=[3, 4, 5])
    parser.add_argument('--num_filters', type=list, default=128)
    parser.add_argument('--l2_reg_lambda', type=float, default=0.0)

    # parser.add_argument('--train_data', type=str,
    #                     default='viewfs://hadoop-meituan/ghnn01/user/hadoop-poistar/jiangfeng/train.csv')
    # parser.add_argument('--valid_data', type=str,
    #                     default='viewfs://hadoop-meituan/ghnn01/user/hadoop-poistar/jiangfeng/valid.csv')
    # parser.add_argument('--tensorboard_dir', type=str,
    #                     default='viewfs://hadoop-meituan/ghnn01/user/hadoop-poistar/jiangfeng/text/tensorboard')
    # parser.add_argument('--save_dir', type=str,
    # default='viewfs://hadoop-meituan/ghnn01/user/hadoop-poistar/jiangfeng/text/')


    parser.add_argument('--test_data', type=str, default='data/multi_test.csv')
    # parser.add_argument('--train_data', type=str, default='data/train.csv')
    parser.add_argument('--train_data', type=str, default='train.csv')
    parser.add_argument('--save_path', type=str, default='cnn_model/model.ckpt')
    parser.add_argument('--tensorboard_dir', type=str, default='model/tensorboard')
    parser.add_argument('--valid_data', type=str, default='data/valid.csv')

    FLAGS, unparser = parser.parse_known_args()

    contents, labels = read_data(FLAGS.train_data)
    words, word_to_id, labels, label_to_id = word_to_id(contents, labels)
    # model = TextRNN(FLAGS.embedding_size, FLAGS.hidden_layers, FLAGS.hidden_units, FLAGS.number_classes,
    #                 FLAGS.learning_rate, FLAGS.sequence_length, FLAGS.vocab_size)
    model = TextCNN(FLAGS.embedding_size, FLAGS.number_classes, FLAGS.sequence_length, FLAGS.learning_rate,
                    FLAGS.filter_sizes, FLAGS.num_filters, FLAGS.vocab_size, FLAGS.l2_reg_lambda)
    # train()
    test()
