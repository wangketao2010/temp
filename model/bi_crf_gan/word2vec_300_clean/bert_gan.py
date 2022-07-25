import tensorflow as tf
from pathlib import Path
from tf_metrics import precision, recall, f1
import os, time, csv
from bert_crf_gan_common import bert_model
from bert_crf_gan_common import Generator_BiLSTM_CRF_Common



class Generator_BiLSTM_CRF_GAN(Generator_BiLSTM_CRF_Common):
    def __init__(self, dropout, num_layers, batch_size, params, filter_sizes, num_filters, dropout_keep_Pro, length,
                 is_training=True,vocab_file=None):
        super().__init__(dropout, num_layers, batch_size, params, filter_sizes, num_filters, dropout_keep_Pro, length,
                 is_training)

        with Path(self.params["tags"]).open( encoding='utf-8') as f:
            self.indices = [idx for idx, tag in enumerate(f)]   # tag 的索引
            self.num_tags = len(self.indices) + 1               # tag 长度 + 1
            print('value of indices is::', self.indices)
        # 转换一个 String 类型的 tensor 到 Int64 的 IDS
        self.vocab_words = tf.contrib.lookup.index_table_from_file(vocab_file,
                                                                   num_oov_buckets=self.params['num_oov_buckets'])
        self.vocab_tags = tf.contrib.lookup.index_table_from_file(self.params['tags'])
        with tf.compat.v1.variable_scope('model', reuse=tf.compat.v1.AUTO_REUSE):
            self.variable_2 = tf.random.normal([self.num_labels, 768], stddev=0.1)              # 17 768 的随机正态分布
            self.variable_2 = tf.Variable(self.variable_2, dtype=tf.float32, trainable=False)   # 创建变量

        self.hidden_unit = self.params['lstm_size']                                             # 隐藏大小 100

    def build_graph(self,bert_config=None):
        self.placeholder()
        self.build_bilstm_crf_gan(bert_config=bert_config)
        self.calculate_loss_op_d()
        self.calculate_loss_op_d_u()
        self.trainstep_op()
        self.init_op()

    def build_bilstm_crf_gan(self, pred_ids=None,bert_config=None):

        self.embedded_chars = bert_model(self.x, is_training=self.is_training,bert_config=bert_config)

        if self.is_training:
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.params['dropout'])

        output = self.embedded_chars
        ###########下面是CRF层，输出为self.pred_ids，即预测的id
        # 全连接层函数，方便了开发者自己手动构造权重矩阵W WW和偏移矩阵 b bb，利用矩阵乘法实现全连接层。
        logits = tf.layers.dense(output, self.num_tags)  # 17
        with tf.compat.v1.variable_scope('crf_param', reuse=tf.compat.v1.AUTO_REUSE):
            self.crf_params = tf.compat.v1.get_variable("crf", [self.num_tags, self.num_tags], dtype=tf.float32)

        # 返回一个表示每个单元的前N个位置的mask张量
        #self.weights = tf.sequence_mask(self.sequence_length)
        # decode_tags:一个形状为[batch_size, max_seq_len] 的tensor，类型是tf.int32，表示最好的序列标记。
        # best_score: 一个形状为[batch_size] 的tensor，包含每个序列解码标签的分数。
        self.pred_ids, self.best_score = tf.contrib.crf.crf_decode(logits, self.crf_params, self.sequence_length)

        self.tags_ids = self.tags
        #tags = self.tags

        pred_ids = tf.nn.embedding_lookup(self.variable_2, self.pred_ids)
        tags = tf.nn.embedding_lookup(self.variable_2, self.tags)
        #output_for_disc = output * pred_ids  #########batch x sentence x embedding
        #output_for_disc_1 = output * tags
        #samples = output_for_disc
        #samples_1 = output_for_disc_1
        self.score = self.discriminator(output * pred_ids)
        self.score_1 = self.discriminator(output * tags)

        # 貌似没什么用了
        m = tf.keras.metrics.Precision()
        m1 = tf.keras.metrics.Recall()
        m.update_state(self.tags_ids, self.pred_ids)
        m1.update_state(self.tags_ids, self.pred_ids)

        self.metrics = {
            'acc': tf.compat.v1.metrics.accuracy(self.tags_ids, self.pred_ids),
            'precision': precision(self.tags_ids, self.pred_ids, self.num_tags, self.indices, average='weighted'),
            'recall': recall(self.tags_ids, self.pred_ids, self.num_tags, self.indices, average='weighted'),
            'f1': f1(self.tags_ids, self.pred_ids, self.num_tags, self.indices, average='weighted'),
        }

        self.log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, self.tags, self.sequence_length,self.crf_params)
        self.loss_g = tf.reduce_mean(-self.log_likelihood)

        self.loss_g_u = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score, labels=self.y)  # 损失函数
        # self.loss_g_u = tf.nn.softmax_cross_entropy_with_logits(logits=self.score, labels=self.y)  # 损失函数
        self.loss_g_u = tf.reduce_mean(self.loss_g_u)

    def calculate_loss_op_d(self):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score_1, labels=self.y)
        # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.score_1, labels=self.y)
        self.loss_d = tf.reduce_mean(losses * 100)

    def calculate_loss_op_d_u(self):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score, labels=self.y)
        # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.score, labels=self.y)
        self.loss_d_u = tf.reduce_mean(losses * 100)

    def trainstep_op(self):
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        optim_g = tf.compat.v1.train.AdamOptimizer(0.0001)
        optim_d = tf.compat.v1.train.AdamOptimizer(1e-4)

        # 更新 bert 的 output层和crf
        print('=====================全部变量======================')
        print(tf.compat.v1.trainable_variables())

        print('====================loss1中更新的权重=======================')
        self.params_g = [param for param in tf.compat.v1.trainable_variables() if
                         ('crf_param' in param.name or 'output' in param.name)
                         and 'attention' not in param.name
                         and 'discriminator_1' not in param.name
                         and 'discriminator' not in param.name
                         and 'embeddings' not in param.name]
        print(self.params_g)
        grads_and_vars = optim_g.compute_gradients(self.loss_g, self.params_g)

        print('=====================loss2中更新的权重======================')
        self.params_g_u = [param for param in tf.compat.v1.trainable_variables() if
                         ('crf_param' not in param.name and 'bert/encoder/layer_11/output' in param.name)]
        print(self.params_g_u)
        grads_and_vars_g_u = optim_g.compute_gradients(self.loss_g_u, self.params_g_u)

        print('======================loss_d and loss_d_u 更新的权重=====================')
        self.params = [param for param in tf.compat.v1.trainable_variables() if ('discriminator' in param.name)]
        print(self.params)
        print('================================')
        grads_and_vars_d = optim_d.compute_gradients(self.loss_d, self.params)
        grads_and_vars_d_u = optim_d.compute_gradients(self.loss_d_u, self.params)

        grads_and_vars_clip = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars]
        grads_and_vars_clip_g_u = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars_g_u]
        grads_and_vars_clip_d = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars_d]
        grads_and_vars_clip_d_u = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars_d_u]

        self.train_op = optim_g.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
        self.train_op_g_u = optim_g.apply_gradients(grads_and_vars_clip_g_u, global_step=self.global_step)
        self.train_op_d = optim_d.apply_gradients(grads_and_vars_clip_d, global_step=self.global_step)
        self.train_op_d_u = optim_d.apply_gradients(grads_and_vars_clip_d_u, global_step=self.global_step)

    def linear(self, input_, output_size, scope=None):

        shape = input_.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]

        # Now the computation.
        with tf.compat.v1.variable_scope("discriminator_1", reuse=tf.compat.v1.AUTO_REUSE):
            matrix = tf.compat.v1.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
            bias_term = tf.compat.v1.get_variable("Bias", [output_size], dtype=input_.dtype)

        return tf.matmul(input_, tf.transpose(matrix)) + bias_term

    def highway(self, input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """

        with tf.compat.v1.variable_scope(scope):
            for idx in range(num_layers):
                g = f(self.linear(input_, size, scope='highway_lin_%d' % idx))

                t = tf.sigmoid(self.linear(input_, size, scope='highway_gate_%d' % idx) + bias)

                output = t * g + (1. - t) * input_
                input_ = output

        return output

    def discriminator(self, x):
        l2_loss = tf.constant(0.0)
        with tf.compat.v1.variable_scope('discriminator', reuse=tf.compat.v1.AUTO_REUSE):
            self.embedded_chars = x
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            pooled_outputs = []
            for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, 768, 1, num_filter]
                    self.W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        self.W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.reduce_max(h, axis=1, keep_dims=True)
                    pooled_outputs.append(pooled)

                    # Combine all the pooled features
            num_filters_total = sum(self.num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                self.h_highway = self.highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.random.truncated_normal([num_filters_total, 2], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                scores = tf.compat.v1.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                #ypred_for_auc = tf.nn.softmax(scores)
                #     self.predictions = tf.argmax(self.scores, 1, name="predictions")
                return scores

    def train_for_unlabeled(self, sess, seqs, seqs_len, labels, tags, max_len):  # self.tags self.max_len
        feed = {self.x: seqs, self.sequence_length: seqs_len, self.tags: labels, self.y: tags, self.max_len: max_len}
        _, loss_train, step_num_, W = sess.run([self.train_op_g_u, self.loss_g_u, self.global_step, self.W], feed_dict=feed)
        return loss_train  # loss_g_u

    def train_for_discri_labeled(self, sess, seqs, seqs_len, labels, tags, max_len):  # sequence_length self.max_len
        feed = {self.x: seqs, self.sequence_length: seqs_len, self.tags: labels, self.y: tags, self.max_len: max_len}
        _, loss_disc_1, step_num = sess.run([self.train_op_d, self.loss_d, self.global_step], feed_dict=feed)
        return loss_disc_1  # loss_d

    def train_for_discri_unlabeled(self, sess, epoch, seqs, seqs_len, labels, tags, max_len):    # self.tags self.max_len
        feed = {self.x: seqs, self.sequence_length: seqs_len, self.tags: labels, self.y: tags, self.max_len: max_len}
        _, loss_disc_1, step_num = sess.run([self.train_op_d_u, self.loss_d_u, self.global_step], feed_dict=feed)
        return loss_disc_1  # loss_d_u
