# coding=utf-8
from loaddata.common import paddingkey
import torch.nn
import torch.nn as nn
import  torch.nn.functional as F
from torch.autograd import Variable
from state import state_instance
import torch.nn.init as init
import numpy as np
import time
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

        if self.args.use_cuda is True:
            self.linear = nn.Linear(in_features=self.args.hidden_size,
                                    out_features=self.args.label_size, bias=False).cuda()
            self.non_linear = nn.Linear(in_features=self.args.rnn_hidden_dim * 2, out_features=self.args.hidden_size,
                                        bias=True).cuda()
        else:
            self.linear = nn.Linear(in_features=self.args.hidden_size,
                                    out_features=self.args.label_size, bias=False)
            self.non_linear = nn.Linear(in_features=self.args.rnn_hidden_dim * 2, out_features=self.args.hidden_size,
                                        bias=True)

        self.dropout = nn.Dropout(self.args.dropout)

        init.xavier_uniform(self.linear.weight)
        init.xavier_uniform(self.non_linear.weight)
        self.non_linear.bias.data.uniform_(-np.sqrt(6 / (self.args.hidden_size + 1)),
                                           np.sqrt(6 / (self.args.hidden_size + 1)))

        self.time_all = []

    def forward(self, features, encoder_out, train=False):

        non_linear = F.tanh(self.non_linear(encoder_out))
        non_linear = self.dropout(non_linear)
        decoder_out = self.linear(non_linear)

        decoder_out_acc = decoder_out
        decoder_out = decoder_out.view(features.batch_length * encoder_out.size(1), -1)

        # for id_batch in range(features.batch_length):
        #     decoder_out_acc[id_batch].data[0][self.args.create_alphabet.appID] = -10e+99

        # prediction result
        # time_all = []
        start_time = time.time()
        batch_state = []
        # @time
        for id_batch in range(features.batch_length):
            feature = features.inst[id_batch]
            state = state_instance(feature)
            real_char_num = feature.chars_size
            for id_char in range(real_char_num):
                if id_char == 0:
                    decoder_out_acc[id_batch].data[0][self.args.create_alphabet.appID] = -10e+99
                self.action(state, id_char, decoder_out_acc[id_batch][id_char], train)
            batch_state.append(state)
        end_time = time.time()
        self.time_all.append(end_time - start_time)
        print(sum(self.time_all))
        # return decoder_out, decoder_out_acc
        return decoder_out, batch_state

    def action(self, state, index, output, train):
        # print("executing action")
        if train:
            action = state.gold[index]
        else:
            actionID = self.getMaxindex(self.args, output)
            action = self.args.create_alphabet.label_alphabet.from_id(actionID)
        state.actions.append(action)

        pos = action.find("#")
        if pos == -1:
            # app
            state.words[-1] += state.chars[index]
        else:
            temp_word = state.chars[index]
            state.words.append(temp_word)
            posLabel = action[pos + 1:]
            state.pos_labels.append(posLabel)
            posId = self.args.create_alphabet.pos_alphabet.loadWord2idAndId2Word(posLabel)
            state.pos_id.append(posId)

    def getMaxindex(self, args, decode_out_acc):
        # print("get max index ......")
        decode_out_acc_list = decode_out_acc.data.tolist()
        maxIndex = decode_out_acc_list.index(np.max(decode_out_acc_list))
        return maxIndex

    def getMaxindex_v1(self, args, decoder_output):
        max = decoder_output.data[0]
        maxIndex = 0
        for idx in range(1, args.label_size):
            if decoder_output.data[idx] > max:
                max = decoder_output.data[idx]
                maxIndex = idx
        return maxIndex


