import sys, pickle, os, random
import numpy as np
import tensorflow as tf

NONE = "O"


def read_corpus(corpus_path):
    return read_corpus_common(corpus_path)


def read_corpus_unlabel(corpus_path):
    return read_corpus_common(corpus_path)


# 把BIO格式的单列文字，根据 \n 读成句子
def read_corpus_common(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


# tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6, "UNK": 7
             }


# 根据不重复字对应的id，将字列表转换成id
def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []  # 需要返回的id列表
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


# 读取之前存储的模型文件 每个不重复的字和他对应的索引
def read_dictionary():
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join("/kaggle/working/temp/Data", "word2id.pkl")
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def batch_yield(data, batch_size, shuffle=False):
    """
    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)
    vocab = read_dictionary()
    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)  # 根据不重复字对应的id，将字列表转换成id
        label_ = [tag2label[tag] for tag in tag_] #把tag转换成对应的数字id
        # sent_ = sent_
        # label_ = tag_

        # 取到 batch_size 个，就停下来，在下一个yield 返回
        if len(seqs) == batch_size:
            if len(seqs) != len(labels):
                print('length of sequence is not equal to length of labels')
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)
    if len(seqs) != 0:
        yield seqs, labels


def batch_yield_with_tag(data, batch_size, shuffle=False):
    return batch_yield_with_tag_common(data,batch_size,1,shuffle)


def batch_yield_with_tag_reverse(data, batch_size, shuffle=False):
    return batch_yield_with_tag_common(data,batch_size,None,shuffle)


def batch_yield_with_tag_common(data, batch_size, direct,shuffle=False):
    if shuffle:
        random.shuffle(data)

    vocab = read_dictionary()
    seqs, labels, tags = [], [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)  # 根据不重复字对应的id，将字列表转换成id
        label_ = [tag2label[tag] for tag in tag_] #把tag转换成对应的数字id
        # sent_ = sent_
        # label_ = tag_

        # 取到 batch_size 个，就停下来，在下一个yield 返回
        if len(seqs) == batch_size:
            if len(seqs) != len(labels):
                print('length of sequence is not equal to length of labels')
            yield seqs, labels, tags
            seqs, labels, tags = [], [], []

        seqs.append(sent_)
        labels.append(label_)
        if direct:
            tags.append([0, 1])
        else:
            tags.append([1, 0])

    if len(seqs) != 0:
        yield seqs, labels, tags


# 把一批句子按照最长句子对齐
def pad_sequences(sequences, pad_mark=None):
    max_len = max(map(lambda x: len(x), sequences))
    if max_len > 512:
        max_len = 512
    seq_list, seq_len_list = [], []
    for seq in sequences:  #####['在', '京', '城', '畅', '饮', '故', '乡', '名', '茶', '，', '一', '缕', '乡', '愁', '随', '香', '气', '溢', '出', '，', '一', '怀', '往', '事', '随', '茶', '而', '至', '，', '细', '细', '品', '来', '，', '又', '仿', '佛', '品', '出', '了', '人', '生', '的', '酸', '甜', '苦', '辣', '…', '…']
        seq = list(seq)
        # seq_ = ["[CLS]"] + seq[:max_len] + ["[SEP]"] + [pad_mark] * max(max_len - len(seq), 0)
        seq_ = seq[:max_len] + [0] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list, max_len


def get_chunks(seq, tags, sess):
    #    print(tags)
    #    NONE = "O"
    #    tags =dict(tags)
    #    default = tags.__getitem__(NONE)
    #    with sess as sess_1:
    #    tf.tables_initializer().run()
    #        tags = tags.eval()
    #    for i in seq:
    #        print(i)
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            #            print(tok)
            #            print('-------------------')
            #            for i in idx_to_tag.items():
            #                print(i)
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag, sess)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def get_chunk_type(tok, idx_to_tag, sess):
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


if __name__ == '__main__':
    corpus_path = './ccks_data_path/train_data1'
    data = read_corpus(corpus_path)
