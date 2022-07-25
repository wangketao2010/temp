import numpy as np

from data import batch_yield, pad_sequences


def get_feed_dict(seqs, labels, pad_mark):
    seqs, seqs_len, max_len = pad_sequences(seqs, pad_mark)

    labels, _, _ = pad_sequences(labels, pad_mark='O')
    return seqs, seqs_len, labels, max_len


# def get_feed_dict_for_unlabel(seqs, labels, pad_mark='.'):
#     seqs, seqs_len, max_len = pad_sequences(seqs, pad_mark)
#     labels, _, _ = pad_sequences(labels, pad_mark='O')
#     return seqs, seqs_len, labels, max_len


def eval_dev(sess, gan, dev):  # , test_size, batch_size, flag=0
    value_lis = []
    medi_lis = []
    metric_lis = gan.evaluate_ori(sess, dev)    # , test_size, batch_size, flag=0
    for ele in metric_lis:                  # test_size ➗ batch_size 个 dic 的 list 内部是实体{acc,precision,recall,f1}
        value_lis.append(ele.values())      # 大小和 metric_lis 基本一样的 List 内部是 List 元素依次是 [acc的值,precision的值,recall的值,f1的值]

    value_lis_transform = zip(*value_lis)   # 把评价值列表按类型分割列表
    for ele in value_lis_transform:         # 依次迭代 acc,precision,recall,f1
        transfor_ele = zip(*ele)            # 由于以上参数都是成对出现的，所以继续拆分
        for ele in transfor_ele:            # 依次迭代 指标的第一部分，指标的第二部分
            medi_lis.append(np.mean(ele))   # 依次是 acc.1 的平均值；acc.2 的平均值；p.1 的平均；p.2 的平均 …… 共八个

    return medi_lis


def eval_dev_eval_bl_eval_dev(epoch_index, index, sess, gan, dev, batch_size, batches_labeled_list, run_once_train,
                           test_data, num_batches, saver):
    print('index == {} ,第  {} epoch 1 eval_dev 开始'.format(index, epoch_index + 1))
    medi_lis = eval_dev(sess, gan, dev)  # , test_size, batch_size, flag=0
    print_metrics(index, epoch_index, medi_lis)

    print('index == {} ,第  {} epoch {} 个 batches_labeled_list 进行 run_once_train 开始'.format(index, epoch_index + 1,len(batches_labeled_list)))
    for i, (words, labels) in enumerate(batches_labeled_list):
        run_once_train(sess, words, labels, tags=[], epoch=epoch_index, gan=gan,
                       num_batches=num_batches, batch=i, label=0, saver=saver)

    dev1 = batch_yield(test_data, batch_size, shuffle=True)
    print('index == {} ,第  {} epoch 2 eval_dev 开始'.format(index, epoch_index + 1))
    medi_lis_from_cross_entropy_training = eval_dev(sess, gan, dev1)  # , test_size, batch_size, flag=0
    print_metrics(index,epoch_index,medi_lis_from_cross_entropy_training)

    print('index == {} ,第  {} epoch eval_dev_eval_bl_eval_dev 结束!!!!!!!!!!!!!!!!!!1'.format(index, epoch_index + 1))


def print_metrics(index, epoch_index, list):
    print('print_metrics index == {} ,第  {} epoch print_metrics  打印:'.format(index, epoch_index + 1))
    print('print_metrics |--acc.1-|-acc.2-|-precision.1-|-precision.2-|-recall.1-|-recall.2-|-f1.1-|-f1.2--|')
    print("print_metrics {}".format(list))
    #print("\n")
