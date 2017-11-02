#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
from loaddata.Load_external_word_embedding import Word_Embedding
from loaddata.Dataloader import load_data, instance
from loaddata.Alphabet import Create_Alphabet, Alphabet
from loaddata.Batch_Iterator import Iterators
import train_ALL_LSTM
import multiprocessing as mu
import shutil
import random
import hyperparams
import hyperparams as hy
import time
# solve encoding
from imp import reload
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# init hyperparams instance
hyperparams = hyperparams.Hyperparams()

# random seed
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)

parser = argparse.ArgumentParser(description="word seg and pos tag")
# learning
parser.add_argument('-lr', type=float, default=hyperparams.learning_rate, help='initial learning rate [default: 0.001]')
parser.add_argument('-learning_rate_decay', type=float, default=hyperparams.learning_rate_decay, help='initial learning_rate_decay rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=hyperparams.epochs, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=hyperparams.batch_size, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=hyperparams.log_interval,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=hyperparams.test_interval, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=hyperparams.save_interval, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default=hyperparams.save_dir, help='where to save the snapshot')
# data path
parser.add_argument('-train_path', type=str, default=hyperparams.train_path, help='train data path')
parser.add_argument('-dev_path', type=str, default=hyperparams.dev_path, help='dev data path')
parser.add_argument('-test_path', type=str, default=hyperparams.test_path, help='test data path')
# shuffle data
parser.add_argument('-shuffle', action='store_true', default=hyperparams.shuffle, help='shuffle the data when load data' )
parser.add_argument('-epochs_shuffle', action='store_true', default=hyperparams.epochs_shuffle, help='shuffle the data every epoch' )
# optim select
parser.add_argument('-Adam', action='store_true', default=hyperparams.Adam, help='whether to select Adam to train')
parser.add_argument('-SGD', action='store_true', default=hyperparams.SGD, help='whether to select SGD to train')
parser.add_argument('-Adadelta', action='store_true', default=hyperparams.Adadelta, help='whether to select Adadelta to train')
parser.add_argument('-momentum_value', type=float, default=hyperparams.optim_momentum_value, help='value of momentum in SGD')
# model
parser.add_argument('-rm_model', action='store_true', default=hyperparams.rm_model, help='whether to delete the model after test acc so that to save space')
parser.add_argument('-init_weight', action='store_true', default=hyperparams.init_weight, help='init w')
parser.add_argument('-init_weight_value', type=float, default=hyperparams.init_weight_value, help='value of init w')
parser.add_argument('-init_weight_decay', type=float, default=hyperparams.weight_decay, help='value of init L2 weight_decay')
parser.add_argument('-init_clip_max_norm', type=float, default=hyperparams.clip_max_norm, help='value of init clip_max_norm')
parser.add_argument('-dropout', type=float, default=hyperparams.dropout, help='the probability for dropout [default: 0.5]')
parser.add_argument('-dropout_embed', type=float, default=hyperparams.dropout_embed, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=hyperparams.max_norm, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=hyperparams.embed_dim, help='number of embedding dimension [default: 128]')
parser.add_argument('-static', action='store_true', default=hyperparams.static, help='fix the embedding')
parser.add_argument('-BiLSTM_1', action='store_true', default=hyperparams.BiLSTM_1, help='whether to use BiLSTM_1 model')
parser.add_argument('-LSTM', action='store_true', default=hyperparams.LSTM, help='whether to use LSTM model')
parser.add_argument('-fix_Embedding', action='store_true', default=hyperparams.fix_Embedding, help='whether to fix word embedding during training')
parser.add_argument('-word_Embedding', action='store_true', default=hyperparams.word_Embedding, help='whether to load word embedding')
parser.add_argument('-word_Embedding_Path', type=str, default=hyperparams.word_Embedding_Path, help='filename of model snapshot [default: None]')
parser.add_argument('-rnn_hidden_dim', type=int, default=hyperparams.rnn_hidden_dim, help='the number of embedding dimension in LSTM hidden layer')
parser.add_argument('-rnn_num_layers', type=int, default=hyperparams.rnn_num_layers, help='the number of embedding dimension in LSTM hidden layer')
parser.add_argument('-min_freq', type=int, default=hyperparams.min_freq, help='min freq to include during built the vocab')
# seed number
parser.add_argument('-seed_num', type=float, default=hy.seed_num, help='value of init seed number')
# nums of threads
parser.add_argument('-num_threads', type=int, default=hyperparams.num_threads, help='the num of threads')
# device
parser.add_argument('-gpu_device', type=int, default=hyperparams.gpu_device, help='gpu number that usr in training')
parser.add_argument('-use_cuda', action='store_true', default=hyperparams.use_cuda, help='disable the gpu')
# option
args = parser.parse_args()

load_data = load_data()
train_data = load_data.loaddate(path=args.train_path, shuffle=True)
dev_data = load_data.loaddate(path=args.dev_path, shuffle=True)
test_data = load_data.loaddate(path=args.test_path, shuffle=True)
create_alphabet = Create_Alphabet(min_freq=1)
create_alphabet.createAlphabet(train_data=train_data, dev_data=dev_data, test_data=test_data)
iter = Iterators(batch_size=args.batch_size, examples=train_data, operator=create_alphabet)
# print(create_alphabet.word_state)
# print(create_alphabet.bichar_alphabet.words2id)
# print(create_alphabet.bichar_alphabet.m_size)


# update parameters
args.cuda = (args.use_cuda) and torch.cuda.is_available(); del args.use_cuda
# save file
mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
args.mulu = mulu
args.save_dir = os.path.join(args.save_dir, mulu)
if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)


# print parameters
print("\nParameters:")
if os.path.exists("./Parameters.txt"):
    os.remove("./Parameters.txt")
file = open("Parameters.txt", "a")
for attr, value in sorted(args.__dict__.items()):
    # if attr.upper() != "PRETRAINED_WEIGHT" and attr.upper() != "pretrained_weight_static".upper():
    print("\t{}={}".format(attr.upper(), value))
    file.write("\t{}={}\n".format(attr.upper(), value))
file.close()
shutil.copy("./Parameters.txt", "./snapshot/" + mulu + "/Parameters.txt")
shutil.copy("./hyperparams.py", "./snapshot/" + mulu)
