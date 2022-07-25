from bert_crf import Generator_BiLSTM_CRF
import os, time, sys
import tensorflow as tf
from data import batch_yield, read_corpus, read_corpus_unlabel, \
    batch_yield_with_tag, batch_yield_with_tag_reverse
from bert_base.bert import modeling
from common import get_feed_dict, eval_dev_eval_bl_eval_dev
from china_main_common import batch_size, params, filter_sizes, num_filters, init_checkpoint, args, model_path, \
    bert_config, vocab_file, dis_dropout_keep_prob, epoch_num


# def train(sess, train, dev, epoch, gen, num_batches, batch, label):
#     """
#     :param train:
#     :param dev:
#     :return:
#     """
#     saver = tf.train.Saver(tf.compat.v1.global_variables())
#
#     run_once_train(sess, train, dev, epoch, saver, gen, num_batches, batch, label)


def run_once_train(sess, words, labels, tags,  epoch, gen, num_batches, batch, label, it, iteration, saver):
    """
    :param sess:
    :param train:
    :param dev:
    :param tag2label:
    :param epoch:
    :param saver:
    :return:
    """
    #   num_batches = (len(train) + batch_size - 1) // batch_size

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if label == 0:
        seqs, labels = words, labels
        #        batches = batch_yield(train, batch_size, shuffle=True)
        #       for step, (seqs, labels) in batch:
        sys.stdout.write(' processing: epoch {} : {} batch / {} batches.'.format(epoch + 1, batch + 1, num_batches) + '\r')
        step_num = epoch * num_batches + batch + 1
        seqs, seqs_len, labels, max_len = get_feed_dict(seqs, labels,pad_mark='.')
        loss_train = gen.train(sess, seqs, seqs_len, labels, max_len)
        print(loss_train)
        print('11111111111111, training_phase_1 finished!')


def main():
    # if args.mode == 'train'
    ap = []
    with open('../../../../china_medical_char_data_cleaned/vocab.tags.txt', 'r', encoding='utf-8') as fin:
        for line in fin:
            ap.append(line.strip())
        fin.close()
    length = len(ap)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.625)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=tf.ConfigProto(
        #         device_count={ "CPU": 48 },
        #         inter_op_parallelism_threads=10,
        allow_soft_placement=True,
        #         intra_op_parallelism_threads=20,
        gpu_options=gpu_options))

    generator = Generator_BiLSTM_CRF(params["dropout"], 1, batch_size, params, filter_sizes, num_filters, dis_dropout_keep_prob, length,vocab_file=vocab_file)
    generator.build_graph(bert_config=bert_config)

    tvars = tf.compat.v1.trainable_variables()
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
        tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    # 最后初始化变量
    # sess.run(tf.compat.v1.global_variables_initializer())

    sess.run(generator.init_op)
    sess.run(generator.table_op)
    sess.run(generator.init_op_1)
    saver = tf.train.Saver(tf.compat.v1.global_variables())

    tf.compat.v1.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        print("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

    train_path = os.path.join('.', args.train_data, 'train_data') # train_data.txt
    train_unlabel_path = os.path.join('.', args.train_data_unlabel,'train_data' ) # train_data.txt
    # train_unlabel_path_1 = os.path.join('.', args.train_data_unlabel, 'train_unlabel1')
    test_path = os.path.join('.', args.test_data, 'test_data')
    sub_test_path = os.path.join('.', args.sub_test_data, 'test_data')
    train_data = read_corpus(train_path)
    train_data_unlabel = read_corpus_unlabel(train_unlabel_path)
    # train_data_unlabel_1 = read_corpus_unlabel(train_unlabel_path_1)
    test_data = read_corpus(test_path);
    test_size = len(test_data)
    sub_test_data = read_corpus(sub_test_path)

    batches_labeled_list = batch_yield(train_data, batch_size, shuffle=True)
    batches_labeled_list = list(batches_labeled_list)
    # print(len(batches_labeled_list))
    num_batches = (len(train_data) + batch_size - 1) // batch_size
    batches_unlabeled_list = batch_yield_with_tag(train_data_unlabel, batch_size, shuffle=True)
    batches_unlabeled_list = list(batches_unlabeled_list)
    # print(len(batches_unlabeled_list))
    batches_labeled_d_list = batch_yield_with_tag(train_data, batch_size, shuffle=True)
    batches_labeled_d_list = list(batches_labeled_d_list)
    batches_unlabeled_d_list = batch_yield_with_tag_reverse(train_data_unlabel, batch_size, shuffle=True)
    batches_unlabeled_d_list = list(batches_unlabeled_d_list)
    dev = batch_yield(test_data, batch_size, shuffle=True)
    #    num_batches = min(len(batches_labeled_list),len(batches_unlabeled_list))
    num_batches_unlabel = (len(train_data_unlabel) + batch_size - 1) // batch_size
    num_batches_1 = min(len(batches_labeled_d_list), len(batches_unlabeled_d_list))
    index = 0
    if args.mode == 'train':
        for epoch_index in range(epoch_num):

            eval_dev_eval_bl_eval_dev(epoch_index,index,sess,generator,dev,batch_size,batches_labeled_list,run_once_train,test_data,num_batches,saver)

    if args.mode == 'test':
        sub_dev = batch_yield_with_tag_reverse(sub_test_data, batch_size, shuffle=True)
        #          print(list(sub_dev))
        ckpt_file = tf.train.latest_checkpoint(model_path)

        generator = Generator_BiLSTM_CRF(params["dropout"], batch_size, params, filter_sizes, num_filters, dis_dropout_keep_prob, length,
                                         is_training=False)
        generator.build_graph(bert_config=bert_config)
        generator.test(sess, sub_dev, test_size, 20)


if __name__ == '__main__':
    # if args.mode == 'train':
    main()

