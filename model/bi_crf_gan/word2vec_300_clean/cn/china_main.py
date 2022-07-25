import os
import sys
import time

import tensorflow as tf
from bert_base.bert import modeling

from bert_gan import Generator_BiLSTM_CRF_GAN
from china_main_common import batch_size, params, \
    filter_sizes, num_filters, init_checkpoint, args, bert_config, vocab_file, epoch_num, dis_dropout_keep_prob
from common import get_feed_dict, eval_dev, eval_dev_eval_bl_eval_dev, print_metrics
from data import batch_yield, read_corpus, read_corpus_unlabel, \
    batch_yield_with_tag, batch_yield_with_tag_reverse, tag2label


# def train(sess, train, dev, epoch, gen, num_batches, batch, label):
#     """
#     :param train:
#     :param dev:
#     :return:
#     """
#     saver = tf.train.Saver(tf.compat.v1.global_variables())
#
#     run_once_train(sess, train, dev, epoch, saver, gen, num_batches, batch, label)


def run_once_train(sess, words, labels, tags,  epoch, gan, num_batches, batch, label, saver):
    """
    :param sess:
    :param train:
    :param dev:
    :param tag2label:
    :param epoch: epoch_index
    :param saver:
    :return:
    """

    sys.stdout.write(' processing: epoch_index {} : {} batch / {} batches.'.format(epoch + 1, batch + 1, num_batches) + '\r')
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if label == 0:
        seqs, labels = words, labels
        seqs, seqs_len, labels, max_len = get_feed_dict(seqs, labels, pad_mark='。')
        loss_train = gan.train(sess, seqs, seqs_len, labels, max_len)
    elif label == 1:
        seqs, labels, tags = words, labels, tags
        seqs, seqs_len, labels, max_len = get_feed_dict(seqs, labels, pad_mark='。')
        loss_train = gan.train_for_unlabeled(sess, seqs, seqs_len, labels, tags, max_len)
    elif label == 2:  # labeled training of discriminator
        seqs, labels, tags = words, labels, tags
        seqs, seqs_len, labels, max_len = get_feed_dict(seqs, labels, pad_mark='。')
        loss_train = gan.train_for_discri_labeled(sess, seqs, seqs_len, labels, tags, max_len)
    else:  # unlabeled training of discriminator
        seqs, labels, tags = words, labels, tags
        seqs, seqs_len, labels, max_len = get_feed_dict(seqs, labels, pad_mark='。')
        loss_train = gan.train_for_discri_unlabeled(sess, epoch, seqs, seqs_len, labels, tags, max_len)
    if batch % 100 == 0:
        print('{} run_once_train label == {}, epoch_index {} : {} batch / {} batches finished loss_train '
              'is {}'.format(start_time, label, epoch + 1, batch + 1, num_batches, loss_train))


def main():

    # ap = []  # 所有 tag 的列表
    # with open('../../../../china_medical_char_data_cleaned/vocab.tags.txt', 'r', encoding='utf-8') as fin:
    #     for line in fin:
    #         ap.append(line.strip())
    #     fin.close()
    length = len(tag2label)  # len(ap)  # tag 标签个数
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.625)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        #         device_count={ "CPU": 48 },
        #         inter_op_parallelism_threads=10,
        allow_soft_placement=True,
        #         intra_op_parallelism_threads=20,
        gpu_options=gpu_options))

    gan = Generator_BiLSTM_CRF_GAN(params["dropout"], 1, batch_size, params, filter_sizes, num_filters, dis_dropout_keep_prob, length,
                                         vocab_file=vocab_file)
    gan.build_graph(bert_config=bert_config)

    # 读取所有可训练的参数
    tvars = tf.compat.v1.trainable_variables()
    # 读取检查点，找到可训练参数之前的状态
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    # 根据之前的检查点，初始化变量
    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    sess.run(gan.init_op)         # 初始化全局变量
    sess.run(gan.table_op)        # 初始化表变量
    sess.run(gan.init_op_1)       # 初始化局部变量
    saver = tf.train.Saver(tf.compat.v1.global_variables())       # 保存器用来保存全局变量

    tf.compat.v1.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = "未初始化"
        if var.name in initialized_variable_names:
            init_string = "已从检查点初始化"  # , *INIT_FROM_CKPT*
        print("变量名 name = %s, 形状 shape = %s 是 %s", var.name, var.shape, init_string)

    train_path = os.path.join('.', args.train_data, 'train_data')  # train_data1
    train_unlabeled_path = os.path.join('.', args.train_data_unlabel, 'train_unlabel')  # "'train_unlabel'
    test_path = os.path.join('.', args.test_data, 'test_data')

    # 读成 List<Tuple<Sent,Tag>>
    train_data = read_corpus(train_path)
    train_data_unlabeled = read_corpus_unlabel(train_unlabeled_path)

    # 读成 List<Tuple<Sent,Tag>>
    test_data = read_corpus(test_path)
    test_size = len(test_data)
    # 读成文字枚举
    dev = batch_yield(test_data, batch_size, shuffle=True)

    # 一共的批次
    num_batches = (len(train_data) + batch_size - 1) // batch_size
    num_batches_unlabeled = (len(train_data_unlabeled) + batch_size - 1) // batch_size

    # 读成文字枚举             List<Tuple<List<20 * List<字的id>>,List<20 * List<字对应Tag的id>>>>
    batches_labeled_list = batch_yield(train_data, batch_size, shuffle=True)
    batches_labeled_list = list(batches_labeled_list)

    # 读成文字枚举，带Tag       List<Tuple<List<20 * List<字的id>>,List<20 * List<字对应Tag的id>>,List<20*[0,1]>>>
    batches_unlabeled_list = batch_yield_with_tag(train_data_unlabeled, batch_size, shuffle=True)
    batches_unlabeled_list = list(batches_unlabeled_list)

    # 读成文字枚举，带Tag       List<Tuple<List<20 * List<字的id>>,List<20 * List<字对应Tag的id>>,List<20*[0,1]>>>
    batches_labeled_d_list = batch_yield_with_tag(train_data, batch_size, shuffle=True)
    batches_labeled_d_list = list(batches_labeled_d_list)
    batches_labeled_d_list_len = len(batches_labeled_d_list)

    # 读成文字枚举，带Tag 反着   List<Tuple<List<20 * List<字的id>>,List<20 * List<字对应Tag的id>>,List<20*[1,0]>>>
    batches_unlabeled_d_list = batch_yield_with_tag_reverse(train_data_unlabeled, batch_size, shuffle=True)
    batches_unlabeled_d_list = list(batches_unlabeled_d_list)
    batches_unlabeled_d_list_len = len(batches_unlabeled_d_list)

    if args.mode == 'train':
        for epoch_index in range(epoch_num):
            index = 0  # 变化方式 0 0 0 0 0 0 0 （0 1 2 3 …… ） * 25
            # 进行一次训练
            eval_dev_eval_bl_eval_dev(epoch_index, index, sess, gan, dev, batch_size, batches_labeled_list,
                            run_once_train, test_data, num_batches, saver)

            if epoch_index > 5:
                for (ele, ele2) in zip(enumerate(batches_labeled_d_list), enumerate(batches_unlabeled_d_list)):
                    index += 1
                    run_once_train(sess, ele[1][0], ele[1][1], ele[1][2], epoch=epoch_index, gan=gan,
                                   num_batches=batches_labeled_d_list_len, batch=index, label=2, saver=saver)
                    run_once_train(sess, ele2[1][0], ele2[1][1], ele2[1][2], epoch=epoch_index, gan=gan,
                                   num_batches=batches_unlabeled_d_list_len, batch=index, label=3, saver=saver)

                print('epoch {} / {} the whole dis phaseI finished'.format(epoch_index + 1,epoch_num))
                for it in range(5):
                    for i, (words, labels, tags) in enumerate(batches_unlabeled_list):
                        run_once_train(sess, words, labels, tags=tags, epoch=epoch_index, gan=gan,
                                       num_batches=num_batches_unlabeled, batch=i, label=1, saver=saver)

                dev2 = batch_yield(test_data, batch_size, shuffle=True)
                medi_lis_from_adversarial_training = eval_dev(sess, gan, dev2)  # , test_size, batch_size, flag=0
                print_metrics(index,epoch_index,medi_lis_from_adversarial_training)

            #print('the accuracy after adversarial training of gan finised!!!!!!!!!!!!!!')
            print('epoch_index {} / {} fully finished!'.format(epoch_index + 1,epoch_num))

    if args.mode == 'test':
        sub_test_path = os.path.join('.', args.sub_test_data, 'test_data')
        sub_test_data = read_corpus(sub_test_path)
        sub_dev = batch_yield_with_tag_reverse(sub_test_data, batch_size, shuffle=True)
        gan = Generator_BiLSTM_CRF_GAN(params["dropout"], batch_size, params, filter_sizes, num_filters, dis_dropout_keep_prob, length,
                                             is_training=False, vocab_file=vocab_file)
        gan.build_graph(bert_config=bert_config)
        gan.test(sess, sub_dev, test_size, 20)


if __name__ == '__main__':
    # if args.mode == 'train':
    main()
