# coding=utf-8
from loaddata.common import paddingkey
import torch.nn
import torch.nn as nn
import  torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy
import random
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)

"""
    sequence to sequence Encode model
"""


class Encoder(nn.Module):

    def __init__(self, args):
        print("Encoder model")
        super(Encoder, self).__init__()
        self.args = args

        # random
        self.char_embed = nn.Embedding(self.args.embed_char_num, self.args.embed_char_dim)
        self.bichar_embed = nn.Embedding(self.args.embed_bichar_num, self.args.embed_bichar_dim)
        # fix the word embedding
        self.static_char_embed = nn.Embedding(self.args.embed_char_num, self.args.embed_char_dim)
        self.static_bichar_embed = nn.Embedding(self.args.embed_bichar_num, self.args.embed_bichar_dim)

        self.lstm_left = nn.LSTM(input_size=self.args.hidden_size, hidden_size=self.args.rnn_hidden_dim, bias=True)
        self.lstm_right = nn.LSTM(input_size=self.args.hidden_size, hidden_size=self.args.rnn_hidden_dim, bias=True)

        self.hidden = self.init_hidden_cell(self.args.rnn_num_layers, self.args.batch_size)

        self.dropout = nn.Dropout(self.args.dropout)
        self.dropout_embed = nn.Dropout(self.args.dropout_embed)

        self.input_dim = (self.args.embed_char_dim + self.args.embed_bichar_dim) * 2
        self.liner = nn.Linear(in_features=self.input_dim, out_features=self.args.hidden_size, bias=True)

    def init_hidden_cell(self, num_layers=1, batch_size=1):
        # the first is the hidden h
        # the second is the cell  c
        if self.args.use_cuda is True:
            return (Variable(torch.zeros(num_layers, batch_size, self.args.rnn_hidden_dim)).cuda(),
                    Variable(torch.zeros(num_layers, batch_size, self.args.rnn_hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(num_layers, batch_size, self.args.rnn_hidden_dim)),
                    Variable(torch.zeros(num_layers, batch_size, self.args.rnn_hidden_dim)))

    def forward(self, features):
        print("Encoder forward")
        batch_length = features.batch_length
        char_features_num = features.char_features.size(1)
        # fine tune
        char_features = self.char_embed(features.char_features)
        bichar_left_features = self.bichar_embed(features.bichar_left_features)
        bichar_right_features = self.bichar_embed(features.bichar_right_features)

        # fix the word embedding
        static_char_features = self.static_char_embed(features.char_features)
        static_bichar_l_features = self.static_bichar_embed(features.bichar_left_features)
        static_bichar_r_features = self.static_bichar_embed(features.bichar_right_features)

        # dropout
        char_features = self.dropout_embed(char_features)
        bichar_left_features = self.dropout_embed(bichar_left_features)
        bichar_left_features = self.dropout_embed(bichar_left_features)
        static_char_features = self.dropout_embed(static_char_features)
        static_bichar_l_features = self.dropout_embed(static_bichar_l_features)
        static_bichar_r_features = self.dropout_embed(static_bichar_r_features)

        # left concat
        left_concat = torch.cat((char_features, static_char_features, bichar_left_features, static_bichar_l_features), 2)
        left_concat = left_concat.view(batch_length * char_features_num, self.input_dim)
        # print(left_concat.size())

        # right concat
        right_concat = torch.cat((char_features, static_char_features, bichar_right_features, static_bichar_r_features), 2)
        right_concat = right_concat.view(batch_length * char_features_num, self.input_dim)
        # print(right_concat.size())

        # non-linear
        left_concat = F.tanh(self.liner(left_concat))
        left_concat = left_concat.view(batch_length, char_features_num, self.args.rnn_hidden_dim)
        left_concat = left_concat.permute(1, 0, 2)
        right_concat = F.tanh(self.liner(right_concat))
        right_concat = right_concat.view(batch_length, char_features_num, self.args.rnn_hidden_dim)
        right_concat = right_concat.permute(1, 0, 2)
        # non-linear dropout
        left_concat = self.dropout(left_concat)
        right_concat = self.dropout(right_concat)

        # init hidden cell
        self.hidden = self.init_hidden_cell(self.args.rnn_num_layers, batch_size=batch_length)
        # lstm
        lstm_left_out, _ = self.lstm_left(left_concat, self.hidden)
        lstm_right_out, _ = self.lstm_right(right_concat, self.hidden)

        # print("lstm_left {} lstm_right {}".format(lstm_left_out.size(), lstm_right_out.size()))

        encoder_output = torch.cat((lstm_left_out, lstm_right_out), 2).permute(1, 0, 2)
        # print(encoder_output.size())
        return encoder_output


