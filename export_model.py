#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import argparse
import os
import tensorflow as tf

from data_helper import words_to_dic
from rnn_model import TextRNN


def export_model(sess, model, path, version, char_to_id):  ## 清空导出目录 注意添加version版本信息

    export_path = os.path.join(path, str(version))
    if tf.gfile.IsDirectory(export_path):
        tf.gfile.DeleteRecursively(export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(model.input_x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(model.logits)
    tensor_info_dropout = tf.saved_model.utils.build_tensor_info(model.keep_prob)

    predict_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'sentences': tensor_info_x, "dropout": tensor_info_dropout},
            outputs={'label': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    )

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_label': predict_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature,
        },
        legacy_init_op=legacy_init_op
    )

    builder.save()
    with tf.gfile.GFile(os.path.join(export_path, "char2id.csv"), "w") as file:
        for key, value in char_to_id.iteritems():
            file.write("%s\t%s\n" % (key, value))
    print("Done exporting!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_size', type=int, default=200)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--number_classes', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--sequence_length', type=int, default=200)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--save_path', type=str,
                        default='model/model.ckpt')
    parser.add_argument('--word_file', type=str,
                        default='model/words.csv')
    FLAGS, unparser = parser.parse_known_args()
    model = TextRNN(FLAGS.embedding_size, FLAGS.hidden_layers, FLAGS.hidden_units, FLAGS.number_classes,
                    FLAGS.learning_rate, FLAGS.sequence_length, FLAGS.vocab_size)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=FLAGS.save_path)

    word2id = words_to_dic(FLAGS.word_file)
    export_model(sess=session, model=model, path='.', version=1, char_to_id=word2id)
