import tensorflow as tf
import numpy as np
import os, argparse, time, random
from bilstm_crf import BiLSTM_CRF
from logger import get_logger
from data_process import str2bool, read_corpus, read_dictionary, tag2label, random_embedding

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

"""
输入参数
"""
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data', help='train data source')
parser.add_argument('--test_data', type=str, default='data', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')#小批量的样本数，训练用到多少个batch，即mini-batch的数量
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')#向前和向后传播中所有批次的单次训练迭代
parser.add_argument('--hidden_dim', type=int, default=30, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')#是否使用CRF层
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')#在培训期间更新嵌入
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')#预训练或者随机初始化
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')#在每次迭代之前更改训练数据
parser.add_argument('--mode', type=str, default='test', help='train/test')#
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test')#用途测试和demo的训练模型
args = parser.parse_args()

"""
获取字和索引
"""
word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))#反序列化，返回字和对应的

"""
获取词向量
"""
if args.pretrain_embedding == 'random':
    print("随机初始化")
    embeddings = random_embedding(word2id, args.embedding_dim)#随机初始化的embedding矩阵
    print(embeddings)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')#经过训练的embedding矩阵

"""
获取语料
"""
train_path = os.path.join('.', args.train_data, 'train_data')
test_path = os.path.join('.', args.test_data, 'test_data')
train_data = read_corpus(train_path)
test_data = read_corpus(test_path)
test_size = len(test_data)


"""
路径设置
"""
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', "result", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


"""
模型训练
"""
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

"""
测试
"""
if args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)  #自动获取保存的最新的一次模型
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)
