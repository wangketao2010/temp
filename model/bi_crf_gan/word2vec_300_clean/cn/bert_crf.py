import tensorflow as tf
from pathlib import Path
from data import  pad_sequences
from tf_metrics import precision, recall, f1
import os, time, csv
from bert_crf_gan_common import bert_model
from bert_crf_gan_common import Generator_BiLSTM_CRF_Common


class Generator_BiLSTM_CRF(Generator_BiLSTM_CRF_Common):
    def __init__(self, dropout, num_layers, batch_size, params, filter_sizes, num_filters, dropout_keep_Pro, length,
                 is_training=True,vocab_file=None):
        super().__init__(dropout, num_layers, batch_size, params, filter_sizes, num_filters, dropout_keep_Pro, length,
                 is_training)

        with Path(self.params["tags"]).open( encoding='utf-8') as f:

            self.indices = [idx for idx, tag in enumerate(f)]  # if tag.strip() != 'O']
            self.num_tags = len(self.indices)

            f.seek(0)

            self.tag_dict = dict()
            for idx, tag in enumerate(f):
                self.tag_dict[tag] = idx
            self.tag_dict = {v: k for k, v in self.tag_dict.items()}
        self.vocab_words = tf.contrib.lookup.index_table_from_file(vocab_file,
                                                                   num_oov_buckets=self.params['num_oov_buckets'])
        self.vocab_tags = tf.contrib.lookup.index_table_from_file(self.params['tags'])
        with tf.compat.v1.variable_scope('model', reuse=tf.compat.v1.AUTO_REUSE):
            self.variable_2 = tf.random.normal([self.num_labels, 768], stddev=0.1)
            self.variable_2 = tf.Variable(self.variable_2, dtype=tf.float32, trainable=False)

        self.hidden_unit = self.params['lstm_size']

    def build_graph(self,bert_config=None):
        self.placeholder()
        self.build_bilstm_crf(bert_config=bert_config)
        # self.calculate_loss_op_d()
        # self.calculate_loss_op_d_u()
        self.trainstep_op()
        self.init_op()

    def build_bilstm_crf(self, pred_ids=None,bert_config=None):

        self.embedded_chars = bert_model(self.x, is_training=self.is_training,bert_config=bert_config)

        if self.is_training:
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, 0.5)

        output = self.embedded_chars
        ###########下面是CRF层，输出为self.pred_ids，即预测的id
        #
        logits = tf.layers.dense(output, self.num_tags)
        with tf.compat.v1.variable_scope('crf_param', reuse=tf.compat.v1.AUTO_REUSE):
            self.crf_params = tf.compat.v1.get_variable("crf", [self.num_tags, self.num_tags], dtype=tf.float32)

        self.weights = tf.sequence_mask(self.sequence_length)

        self.pred_ids, self.best_score = tf.contrib.crf.crf_decode(logits, self.crf_params, self.sequence_length)

        # tags = self.vocab_tags.lookup(self.tags) 数据类型错误作出的转换
        self.tags_ids = self.tags
        self.log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, self.tags, self.sequence_length, self.crf_params)
        self.loss_g = tf.reduce_mean(-self.log_likelihood)

        self.score = self.best_score
        self.metrics = {
            'acc': tf.compat.v1.metrics.accuracy(self.tags_ids, self.pred_ids),  # self.weights),
            'precision': precision(self.tags_ids, self.pred_ids, self.num_tags, self.indices),  # self.weights),
            'recall': recall(self.tags_ids, self.pred_ids, self.num_tags, self.indices),  # self.weights),
            'f1': f1(self.tags_ids, self.pred_ids, self.num_tags, self.indices),  # self.weights),
        }

    def trainstep_op(self):
        #  with tf.variable_scope("train_step",reuse=tf.compat.v1.AUTO_REUSE):
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optim = tf.train.AdamOptimizer(0.0001)

        optim1 = tf.train.AdamOptimizer(1e-4)
        #     self.params_g = [param for param in tf.compat.v1.trainable_variables() if
        #                      ('crf_param' and 'output' in param.name)]
        #  self.params_g = [param for param in tf.compat.v1.trainable_variables() if
        #                   ('crf_param' in param.name or 'output' in param.name )  and 'attention' not in param.name and 'discriminator_1' not in param.name and 'discriminator' not in param.name and 'embeddings' not in param.name ]
        self.params_g = [param for param in tf.compat.v1.trainable_variables()]
        print(self.params_g)
        #    self.params_g_u = [param for param in tf.compat.v1.trainable_variables() if
        #                     ('discriminator' not in param.name and 'crf_param' not in param.name and 'output' in param.name)]
        grads_and_vars = optim.compute_gradients(self.loss_g, self.params_g)

        grads_and_vars_clip = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars]

        self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def predict(self, sess, dev):
        seqs = []
        labels_li = []
        labels_lis = []
        for step, (seq, labels, tags) in enumerate(dev):

            #       print('seq is the', seq)
            #       print(('lenght of seq is the', len(seq)))

            seqs, seqs_len, max_len = self.get_feed_dict_seq(seq)
            #       print('seqs is the', seqs)
            #       print('lenght of seqs is the', len(seqs))
            feed = {self.x: seqs, self.sequence_length: seqs_len}
            best_score, pred = sess.run([self.best_score, self.pred_ids], feed_dict=feed)
            #        print(pred)
            for sen in pred:

                for ele in sen:
                    labels_li.append(self.tag_dict[ele])
                #              print('labels is the',labels_li)
                labels_lis.append(labels_li)
                labels_li = []

            if len(labels_lis) != len(seqs):
                print('error, lenght of seqs and labels is not equal!')

            if len(labels_lis) != len(seq):
                print('error, lenght of seqs and labels is not equal!')

            with open('predict_file_baseline.utf8', 'a', encoding='utf-8') as fout:
                for ele in zip(seqs, labels_lis):
                    for elep in zip(*ele):
                        if elep[0] in ['。', ';']:
                            fout.write(elep[0] + '\t' + elep[1] + '\n' + '\n')
                        else:

                            fout.write(elep[0] + '\t' + elep[1] + '\n')
            labels_lis = []

    def get_feed_dict_seq(self, seqs):
        seqs, seqs_len, max_len = pad_sequences(seqs, pad_mark='。')
        return seqs, seqs_len, max_len

    def train_for_unlabeled(self, sess, epoch, seqs, seqs_len, labels, tags, max_len, it, iteration, saver):
        #        self.stack_varia = tf.TensorArray(dtype=tf.float32, size=20, dynamic_size=True)
        feed = {self.x: seqs, self.sequence_length: seqs_len, self.tags: labels, self.y: tags, self.max_len: max_len}
        _, loss_train, step_num_, W = sess.run([self.train_op_g_u, self.loss_g_u, self.global_step, self.W], feed_dict=feed)
        if it == 4 and iteration >= 79:
            # print(W)
            time_stamp = time.time()
            self.model_path = os.path.join('./model', str(int(time_stamp)))
            saver.save(sess, self.model_path, global_step=epoch)  #保存模型
        #    print('variable2 is oooooooooooooooooooooooooooo',vara2)
        return loss_train

