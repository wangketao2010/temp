from bert_gan import Generator_BiLSTM_CRF_GAN
import os, time, sys
import tensorflow as tf
from data import batch_yield, read_corpus, read_corpus_unlabel, \
    batch_yield_with_tag, batch_yield_with_tag_reverse
from bert_base.bert import modeling
from common import get_feed_dict, eval_dev,eval_dev_eval_bl_eval_dev
from en.main_common import batch_size, params, filter_sizes, num_filters, init_checkpoint, args, model_path,vocab_file,bert_config


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
        seqs, seqs_len, labels, max_len = get_feed_dict(seqs, labels, pad_mark='。')
        loss_train = gen.train(sess, seqs, seqs_len, labels, max_len)
        print(loss_train)
        print('11111111111111, training_phase_1 finished!')
    elif label == 1:
        #        batches = batch_yield_with_tag(train, batch_size, shuffle=True)
        #        for step, (seqs, labels,tags) in enumerate(batches):
        seqs, labels, tags = words, labels, tags
        sys.stdout.write(
            ' processing: epoch {} : {} batch / {} batches.'.format(epoch + 1, batch + 1, num_batches) + '\r')
        step_num = epoch * num_batches + batch + 1
        seqs, seqs_len, labels, max_len = get_feed_dict(seqs, labels, pad_mark='。')
        loss_train = gen.train_for_unlabeled(sess, seqs, seqs_len, labels, tags, max_len)
        print(loss_train)
        print('222222222222, training_ohase_II finished!')
    elif label == 2:

        seqs, labels, tags = words, labels, tags

        sys.stdout.write(
            ' processing: epoch {} : {} batch / {} batches.'.format(epoch + 1, batch + 1, num_batches) + '\r')
        step_num = epoch * num_batches + batch + 1
        seqs, seqs_len, labels, max_len = get_feed_dict(seqs, labels, pad_mark='。')
        loss_train = gen.train_for_discri_labeled(sess, seqs, seqs_len, labels, tags, max_len)
        print(loss_train)
        print('333333333333333333333,labeled training of discriminator finised!')
    else:
        seqs, labels, tags = words, labels, tags
        sys.stdout.write(
            ' processing: epoch {} : {} batch / {} batches.'.format(epoch + 1, batch + 1, num_batches) + '\r')
        step_num = epoch * num_batches + batch + 1
        seqs, seqs_len, labels, max_len = get_feed_dict(seqs, labels, pad_mark='。')
        loss_train = gen.train_for_discri_unlabeled(sess, epoch, seqs, seqs_len, labels, tags, max_len)
        print(loss_train)
        print('44444444444444444, unlabeled training of discriminator finised!')


def main():
    # if args.mode == 'train'
    ap = []
    with open('../../../../medical_char_data_cleaned/vocab.tags.txt', 'r', encoding='utf-8') as fin:
        for line in fin:
            ap.append(line.strip())
        fin.close()
    length = len(ap)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.625)
    print(tf.get_default_graph())
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=tf.ConfigProto(
        #         device_count={ "CPU": 48 },
        #         inter_op_parallelism_threads=10,
        allow_soft_placement=True,
        #         intra_op_parallelism_threads=20,
        gpu_options=gpu_options))
    print(sess.graph)
    generator = Generator_BiLSTM_CRF_GAN(0.5, 1, batch_size, params, filter_sizes, num_filters, 0.75, length,vocab_file=vocab_file)
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

    train_path = os.path.join('.', args.train_data, 'train.txt')
    train_unlabel_path = os.path.join('.', args.train_data_unlabel, 'train.txt')
    # train_unlabel_path_1 = os.path.join('.', args.train_data_unlabel, 'train.txt')
    test_path = os.path.join('.', args.test_data, 'test.txt')
    sub_test_path = os.path.join('.', args.sub_test_data, 'test.txt')
    train_data = read_corpus(train_path)
    train_data_unlabel = read_corpus_unlabel(train_unlabel_path)
    # train_data_unlabel_1 = read_corpus_unlabel(train_unlabel_path_1)
    test_data = read_corpus(test_path);
    test_size = len(test_data)
    sub_test_data = read_corpus(sub_test_path)

    batches_labeled_list = batch_yield(train_data, batch_size, shuffle=True)
    batches_labeled_list = list(batches_labeled_list)
    num_batches = (len(train_data) + batch_size - 1) // batch_size
    batches_unlabeled_list = batch_yield_with_tag(train_data_unlabel, batch_size, shuffle=True)
    batches_unlabeled_list = list(batches_unlabeled_list)
    batches_labeled_d_list = batch_yield_with_tag(train_data, batch_size, shuffle=True)
    batches_labeled_d_list = list(batches_labeled_d_list)
    batches_unlabeled_d_list = batch_yield_with_tag_reverse(train_data_unlabel, batch_size, shuffle=True)
    batches_unlabeled_d_list = list(batches_unlabeled_d_list)
    dev = batch_yield(test_data, batch_size, shuffle=True)
    num_batches_unlabel = (len(train_data_unlabel) + batch_size - 1) // batch_size
    # num_batches_1 = min(len(batches_labeled_d_list), len(batches_unlabeled_d_list))
    index = 0
    if args.mode == 'train':

        for epoch_index in range(20):
            # from tensorflow.keras.utils import plot_model
            # plot_model(generator, to_file='model.png', show_shapes=True)  # 保存模型结构图

            eval_dev_eval_bl_eval_dev(epoch_index,index,sess,generator,dev,batch_size,batches_labeled_list,run_once_train,test_data,num_batches,saver)

            if epoch_index > 4:
                #     batches_labeled_d_list = batches_labeled_d_list[0: len(batches_labeled_d_list)-5]
                batches_labeled_d_list_len = len(batches_labeled_d_list)
                batches_unlabeled_d_list_len = len(batches_unlabeled_d_list)
                for (ele, ele2) in zip(enumerate(batches_labeled_d_list), enumerate(batches_unlabeled_d_list)):
                    index += 1
                    run_once_train(sess, ele[1][0], ele[1][1], ele[1][2],  epoch=epoch_index,
                                  gen=generator,
                                  num_batches=batches_labeled_d_list_len, batch=index, label=2, it=0, iteration=0, saver=saver)
                    run_once_train(sess, ele2[1][0], ele2[1][1], ele2[1][2],  epoch=epoch_index,
                                  gen=generator,
                                  num_batches=batches_unlabeled_d_list_len, batch=index, label=3, it=0, iteration=0,
                                  saver=saver)
                index = 0

                print('the whole dis phaseI finished')

                for it in range(5):
                    for i, (words, labels, tags) in enumerate(batches_unlabeled_list):
                        run_once_train(sess, words, labels, tags=tags,  epoch=epoch_index, gen=generator,
                                      num_batches=num_batches_unlabel, batch=i, label=1, it=it, iteration=i,
                                      saver=saver)
                dev2 = batch_yield(test_data, batch_size, shuffle=True)

                medi_lis_from_adversarial_training = eval_dev(sess, generator, dev2)  #, test_size, batch_size, flag=0

                for ele in medi_lis_from_adversarial_training:
                    print('第二次打印', ele)

            print('the accuracy after adversarial training of generator finised!!!!!!!!!!!!!!')

            print('epoch {} finished!'.format(epoch_index + 1))

    if args.mode == 'test':
        sub_dev = batch_yield_with_tag_reverse(sub_test_data, batch_size, shuffle=True)
        #          print(list(sub_dev))
        ckpt_file = tf.train.latest_checkpoint(model_path)

        # print(ckpt_file)
        generator = Generator_BiLSTM_CRF_GAN(0.5, batch_size, params, filter_sizes, num_filters, 0.75, length,
                                             is_training=False)
        generator.build_graph(bert_config=bert_config)
        generator.test(sess, sub_dev, test_size, 20)


if __name__ == '__main__':
    # if args.mode == 'train':
    main()
