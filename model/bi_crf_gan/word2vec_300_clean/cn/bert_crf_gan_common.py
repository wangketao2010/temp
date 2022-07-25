import sys

import numpy as np
import tensorflow as tf
from bert_base.bert import modeling
from tensorflow.contrib.layers.python.layers import initializers
from common import get_feed_dict
from data import pad_sequences


# 中文模型路径



def bert_model(input_ids, is_training,bert_config=None):
    with tf.compat.v1.Session() as sess:
        input_mask = tf.zeros(shape=tf.shape(input_ids), dtype=np.int32)
        token_type_ids = tf.zeros(shape=tf.shape(input_ids), dtype=np.int32)
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
            use_one_hot_embeddings=True
        )

        embeddings = model.get_sequence_output()
        return embeddings


class Generator_BiLSTM_CRF_Common(object):
    def __init__(self, dropout, num_layers, batch_size, params, filter_sizes, num_filters, dropout_keep_Pro, length,
                 is_training=True):
        self.dropout = dropout                          # 逐出率 0.5
        self.num_layers = num_layers                    # 层数 20
        self.batch_size = batch_size                    # 批大小 20
        self.params = params                            # 所有超参
        self.filter_sizes = filter_sizes                # ? [1,2,3,4]
        self.num_filters = num_filters                  # ? [100,200,200,200]
        self.dropout_keep_prob = dropout_keep_Pro       # 0.75
        # self.stack_varia = tf.TensorArray(dtype=tf.float32, size=20, dynamic_size=True)     #
        # self.stack_varia_1 = tf.TensorArray(dtype=tf.float32, size=20, dynamic_size=True)   #
        self.num_labels = length + 1                    # tag 长度 + 1
        self.is_training = is_training                  # 是否训练
        self.initializers = initializers                # 初始化器

    def placeholder(self):
        self.x = tf.compat.v1.placeholder(tf.int32, shape=[None, None])
        self.sequence_length = tf.compat.v1.placeholder(tf.int32, shape=[None])  # batch下序列长度
        self.tags = tf.compat.v1.placeholder(tf.int32, shape=[None, None])  # label
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
        self.max_len = tf.compat.v1.placeholder(tf.int32)
        # self.lr = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="lr")

    def init_op(self):
        # 含有tf.Variable的环境下，因为tf中建立的变量是没有初始化的，也就是在debug时还不是一个tensor量，而是一个Variable变量类型 必须global init
        self.init_op = tf.compat.v1.global_variables_initializer()
        # 一个 Op，初始化图中的所有局部变量
        self.init_op_1 = tf.compat.v1.local_variables_initializer()
        # 函数返回初始化所有表的操作.请注意,如果没有表格,则返回的操作是空操作.
        self.table_op = tf.compat.v1.tables_initializer()

    def evaluate_ori(self, sess, dev):
        metrc_lis = []
        for step, (seq, labels) in enumerate(dev):
            sys.stdout.write('evaluate_ori: running…… step {}  '.format(step + 1) + '\r')
            seqs, seqs_len, labels, _ = get_feed_dict(seq, labels,None)

            feed = {self.x: seqs, self.sequence_length: seqs_len, self.tags: labels}  # self.tags
            best_score, score, metrics, pred = sess.run([self.best_score, self.score, self.metrics, self.pred_ids],
                                                        feed_dict=feed)
            # print('真实结果',labels[1])
            # print('预测结果',pred[1])
            metrc_lis.append(metrics)

        return metrc_lis

    def evaluate(self, sess, dev, test_size, batch_size, flag):
        #        iniit = tf.group(tf.compat.v1.global_variables_initializer(), tf.local_variables_initializer())
        metrc_lis = []
        index = 0
        #    print('dev111111111111: ',list(dev))
        #     print('dev222222222222:', (list(dev))[-1])
        for step, (seq, labels, tags) in enumerate(dev):
            #   index += 1

            index += 1
            seqs, seqs_len, labels, _ = get_feed_dict(seq, labels,None)
            #       print(seqs[1])
            #       print(labels[1])

            #            for i in range(len(seqs)):
            #                print(str(i) + seqs[i])
            feed = {self.x: seqs, self.sequence_length: seqs_len, self.tags: labels, self.y: tags}
            best_score, score, loss, metrics = sess.run([self.best_score, self.score, self.loss_g_u, self.metrics],
                                                        feed_dict=feed)
            print('score is', score)
            metrc_lis.append(metrics)

            if flag == 1:
                #           print('when flag euqal to 1')
                #           print(W)
                #           bess_core_l, scroe_l= [],[]
                #            for i in range(len(best_score)):
                #                    score.append(i)

                with open('./uncertainty_scheme/setence.txt', 'a', encoding='utf-8') as fout:
                    #    besscore = []
                    for i in range(len(seqs)):
                        #            biglist = []
                        besscore = []
                        besscore.append(str(best_score[i]))
                        fout.writelines(p for p in seqs[i])
                        fout.write('\t')
                        fout.writelines(p for p in labels[i])
                        fout.write('\t')
                        fout.writelines(p for p in besscore)
                        fout.write('\t')
                        for p in list(score[i]):
                            #     print(i)
                            fout.write(str(p) + '\t')
                        #             fout.write('\t')
                        fout.write(str(loss))
                        fout.write('\n')
            #     fout.close()

            #  fout.writelines(p for p in list(score[i]))

        #                  fout.write(p for p in besscore)
        #                  fout.write(p for p in list(score[i]))
        #                  Ifout.write('\n')
        #            fout.write(seqs[i] + labels[i] + '\t' +besscore + '\t' + list(score[i]) + '\n')
        print('index is {}'.format(index))
        return metrc_lis

    def test(self, sess, dev, test_size, batch_size):

        #    var_name_list = [v.name for v in tf.compat.v1.trainable_variables()]
        #    for name in var_name_list:
        #        print(name)
        #       tf.reset_default_graph()
        saver = tf.train.Saver()
        with tf.compat.v1.Session() as sess:
            #       tf.reset_default_graph()
            saver.restore(sess, './model/1577729956-2')  # 恢复模型

            #        saver = tf.train.Saver()

            self.evaluate(sess, dev, test_size, batch_size, flag=1)

    # def get_feed_dict(self, seqs, labels):
    #     seqs, seqs_len, max_len = pad_sequences(seqs, pad_mark='O')
    #     labels, _, _ = pad_sequences(labels, pad_mark='O')
    #     return seqs, seqs_len, labels, max_len

    def train(self, sess, seqs, seqs_len, labels, max_len):
        feed = {self.x: seqs, self.sequence_length: seqs_len, self.tags: labels, self.max_len: max_len}  # self.max_len
        _, loss_train, step_num_ = sess.run([self.train_op, self.loss_g, self.global_step], feed_dict=feed)
        #    print(pred[1])i
        return loss_train  # loss_g
