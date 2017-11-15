# coding=utf-8
from loaddata.common import paddingkey
import torch.nn
import torch.nn as nn
import  torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import random
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)

"""
    sequence to sequence Decoder model
"""


class Decoder(nn.Module):

    def __init__(self, args):
        print("Decoder model")
        super(Decoder, self).__init__()
        self.args = args

        self.linear = nn.Linear(in_features=self.args.hidden_size,
                                out_features=self.args.label_size, bias=False)

        self.non_linear = nn.Linear(in_features=self.args.rnn_hidden_dim * 2, out_features=self.args.hidden_size,
                                    bias=True)
        self.dropout = nn.Dropout(self.args.dropout)

        init.xavier_uniform(self.linear.weight)
        init.xavier_uniform(self.non_linear.weight)
        self.non_linear.bias.data.uniform_(-np.sqrt(6 / (self.args.hidden_size + 1)),
                                           np.sqrt(6 / (self.args.hidden_size + 1)))

    def forward(self, features, encoder_out):
        # print(encoder_out.size())
        # print("Decoder forward")

        non_linear = F.tanh(self.non_linear(encoder_out))
        # non_linear = F.softmax(self.non_linear(encoder_out))
        non_linear = self.dropout(non_linear)
        decoder_out = self.linear(non_linear)
        decoder_out_acc = decoder_out
        decoder_out = decoder_out.view(features.batch_length * encoder_out.size(1), -1)
        print("a", decoder_out_acc.size())
        print("b", decoder_out.size())

        return decoder_out, decoder_out_acc


