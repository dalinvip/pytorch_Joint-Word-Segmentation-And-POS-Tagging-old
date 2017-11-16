# coding=utf-8
from loaddata.common import paddingkey
import torch.nn
import torch.nn as nn
import  torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import random
from state import state_instance
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)

"""
    sequence to sequence Decoder model
"""


class Decoder_WordLstm(nn.Module):

    def __init__(self, args):
        print("Decoder model")
        super(Decoder_WordLstm, self).__init__()
        self.args = args

        self.lstm = nn.LSTM(input_size=self.args.hidden_size, hidden_size=self.args.rnn_hidden_dim, bias=True)

        self.pos_embed = nn.Embedding(num_embeddings=self.args.pos_size, embedding_dim=self.args.pos_dim)

        self.linear = nn.Linear(in_features=self.args.rnn_hidden_dim * 2 + self.args.hidden_size,
                                out_features=self.args.label_size, bias=False)

        self.non_linear = nn.Linear(in_features=self.args.rnn_hidden_dim * 2, out_features=self.args.hidden_size,
                                    bias=True)
        self.dropout = nn.Dropout(self.args.dropout)

        init.xavier_uniform(self.linear.weight)
        init.xavier_uniform(self.non_linear.weight)
        self.non_linear.bias.data.uniform_(-np.sqrt(6 / (self.args.hidden_size + 1)),
                                           np.sqrt(6 / (self.args.hidden_size + 1)))

        self.bucket = Variable(torch.zeros(1, self.args.label_size))
        self.bucket_rnn = Variable(torch.zeros(1, self.args.rnn_hidden_dim))
        if self.args.use_cuda is True:
            self.bucket = self.bucket.cuda()
            self.bucket_rnn = self.bucket_rnn.cuda()

    def forward(self, features, encoder_out, train=False):
        # print(encoder_out.size())
        # print("Decoder forward")
        batch_length = features.batch_length
        char_features_num = features.char_features.size(1)
        batch_output = []
        batch_state = []
        for id_batch in range(batch_length):
            feature = features.inst[id_batch]
            state = state_instance(feature)
            sent_output = []
            real_char_num = feature.chars_size
            for id_char in range(char_features_num):
                if id_char < real_char_num:
                    v = torch.cat((self.bucket_rnn, encoder_out[id_batch][id_char].view(1, self.args.rnn_hidden_dim * 2)), 1)
                    # print("232", v.size())
                    output = self.linear(v)
                    if id_char is 0:
                        output.data[0][self.args.create_alphabet.appID] = -1e+99
                    sent_output.append(output)
                else:
                    sent_output.append(self.bucket)
            sent_output = torch.cat(sent_output, 0)
            batch_output.append(sent_output)

        batch_output = torch.cat(batch_output, 0)
        # print("batch_output", batch_output.size())
        decoder_out_acc = batch_output.view(batch_length, encoder_out.size(1), -1)
        # print("de", decoder_out_acc.size())
        return batch_output, decoder_out_acc


