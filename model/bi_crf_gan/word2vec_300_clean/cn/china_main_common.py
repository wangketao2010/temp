import os, time, sys, argparse
from bert_base.bert import modeling

bert_path = os.path.abspath('../../../../files_cn')  # '/home/ywd/tf_model/pre_training_model/chinese_L-12_H-768_A-12/'
init_checkpoint = os.path.join(bert_path, 'bert_model.ckpt')
#################################

# chinese data ccks
data_path = '../../../../china_medical_char_data_cleaned'
# # english data 2010ib
# data_path = 'data_path'
#### Generator Hyper-parameters
batch_size = 20
epoch_num = 10
# filter_sizes = [1, 2, 3, 4, 5, 6]
filter_sizes = [1, 2, 3, 4]
num_filters = [100, 200, 200, 200]
dis_dropout_keep_prob = 0.75
#dis_l2_reg_lambda = 0.2
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='../../../../Data', help='train data source')
parser.add_argument('--train_data_unlabel', type=str, default='../../../../Data', help='train data source')
parser.add_argument('--mode', type=str, default='train', help='train/test')
# parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
parser.add_argument('--test_data', type=str, default='../../../../Data', help='test data source')
parser.add_argument('--sub_test_data', type=str, default='../../../../Data', help='test data source')
args = parser.parse_args()
# train_data =
params = {
    'dim': 768,
    'dropout': 0.5,
    'num_oov_buckets': 1,
    # 'batch_size': 20,
    'buffer': 15000,
    'lstm_size': 100,
    'words': '../../../../china_medical_char_data_cleaned/vocab.words.txt',     # 貌似没用
    'chars': '../../../../china_medical_char_data_cleaned/vocab.chars.txt',     # 貌似没用
    'tags': '../../../../china_medical_char_data_cleaned/vocab.tags.txt',
    'glove': '../../../../china_medical_char_data_cleaned/glove.npz',           # 貌似没用
    'vector': 'bert_vec.npz'                                                    # 貌似没用
}
model_path = './model/'
bert_config = modeling.BertConfig.from_json_file(os.path.join(os.path.abspath('../../../../files_cn'), 'bert_config.json'))
vocab_file = os.path.join(os.path.abspath('../../../../files_cn'), 'vocab.txt')

# parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
# parser.add_argument('--mode', type=str, default='train', help='train/test')
# args = parser.parse_args()

