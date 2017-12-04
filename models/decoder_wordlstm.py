# coding=utf-8
from loaddata.common import paddingkey
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import random
from state import state_instance
from state import state_batch_instance
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

        self.pos_paddingKey = self.args.create_alphabet.pos_PaddingID

        # self.lstm = nn.LSTM(input_size=self.args.hidden_size, hidden_size=self.args.rnn_hidden_dim, bias=True)
        self.lstmcell = nn.LSTMCell(input_size=self.args.hidden_size, hidden_size=self.args.rnn_hidden_dim, bias=True)
        init.xavier_uniform(self.lstmcell.weight_ih)
        init.xavier_uniform(self.lstmcell.weight_hh)
        self.lstmcell.bias_hh.data.uniform_(-np.sqrt(6 / (self.args.rnn_hidden_dim + 1)),
                                            np.sqrt(6 / (self.args.rnn_hidden_dim + 1)))
        self.lstmcell.bias_ih.data.uniform_(-np.sqrt(6 / (self.args.rnn_hidden_dim + 1)),
                                            np.sqrt(6 / (self.args.rnn_hidden_dim + 1)))

        self.pos_embed = nn.Embedding(num_embeddings=self.args.pos_size, embedding_dim=self.args.pos_dim,
                                      padding_idx=self.pos_paddingKey)
        # self.pos_embed = nn.Embedding(num_embeddings=self.args.pos_size, embedding_dim=self.args.pos_dim)
        init.uniform(self.pos_embed.weight,
                     a=-np.sqrt(3 / self.args.pos_dim),
                     b=np.sqrt(3 / self.args.pos_dim))
        self.pos_embed.weight.requires_grad = True

        self.linear = nn.Linear(in_features=self.args.rnn_hidden_dim * 2 + self.args.hidden_size,
                                out_features=self.args.label_size, bias=False)

        # self.non_linear = nn.Linear(in_features=self.args.rnn_hidden_dim * 2, out_features=self.args.hidden_size,
        #                             bias=True)

        self.combine_linear = nn.Linear(in_features=self.args.pos_dim,
                                        out_features=self.args.hidden_size, bias=True)

        init.xavier_uniform(self.linear.weight)
        # init.xavier_uniform(self.non_linear.weight)
        init.xavier_uniform(self.combine_linear.weight)
        # self.non_linear.bias.data.uniform_(-np.sqrt(6 / (self.args.hidden_size + 1)),
        #                                    np.sqrt(6 / (self.args.hidden_size + 1)))
        self.combine_linear.bias.data.uniform_(-np.sqrt(6 / (self.args.hidden_size + 1)),
                                               np.sqrt(6 / (self.args.hidden_size + 1)))

        self.dropout = nn.Dropout(self.args.dropout)

        self.softmax = nn.LogSoftmax()

        self.bucket = Variable(torch.zeros(1, self.args.label_size))
        self.bucket_rnn = Variable(torch.zeros(1, self.args.rnn_hidden_dim))
        if self.args.use_cuda is True:
            self.bucket = self.bucket.cuda()
            self.bucket_rnn = self.bucket_rnn.cuda()

        self.z_bucket = Variable(torch.zeros(16, self.args.hidden_size))
        self.h_bucket = Variable(torch.zeros(16, self.args.rnn_hidden_dim))
        self.c_bucket = Variable(torch.zeros(16, self.args.rnn_hidden_dim))
        if self.args.use_cuda is True:
            self.z_bucket = self.z_bucket.cuda()
            self.h_bucket = self.h_bucket.cuda()
            self.c_bucket = self.c_bucket.cuda()

    def forward(self, features, encoder_out, train=False):

        batch_length = features.batch_length
        # print(encoder_out.size())
        encoder_out = encoder_out.permute(1, 0, 2)
        char_features_num = encoder_out.size(0)
        # print(char_features_num)
        state = state_batch_instance(features, char_features_num)
        # print(len(state.gold))
        # state.show()
        char_output = []
        batch_state = []
        # real_char_num = features.chars_size
        for id_char in range(char_features_num):
            # print("id_char", id_char)
            # if id_char < real_char_num:
            if id_char is 0:
                h, c, z = self.h_bucket, self.c_bucket, self.z_bucket
            else:
                # h, c = state.word_hiddens[-1], state.word_cells[-1]
                h, c = self.h_bucket, self.c_bucket
                last_pos = Variable(torch.zeros(batch_length)).type(torch.LongTensor)
                if self.args.use_cuda is True:
                    last_pos = last_pos.cuda()
                pos_id_array = np.array(state.pos_id[-1])
                last_pos.data.copy_(torch.from_numpy(pos_id_array))
                # print(last_pos)
                last_pos_embed = self.dropout(self.pos_embed(last_pos))
                z = self.dropout(F.tanh(self.combine_linear(last_pos_embed)))
            h_now, c_now = self.lstmcell(z, (h, c))
            v = torch.cat((h_now, encoder_out[id_char]), 1)
            # print(v)
            output = self.linear(v)
            # print(output.size())
            # should cut later write
            if id_char is 0:
                for i in range(batch_length):
                    output.data[i][self.args.create_alphabet.appID] = -10e9
            self.action(state, id_char, output, h_now, c_now, train)
            # print(state.words)
            # print(state.word_hiddens[id_char])

            char_output.append(output.unsqueeze(1))
            batch_state.append(state)
        decoder_out = torch.cat(char_output, 1)
        decoder_out = self.softmax(decoder_out)
        # decoder_out = decoder_out.permute(1, 0, 2).contiguous()
        # print(decoder_out.size())
        decoder_out = decoder_out.view(decoder_out.size(0) * decoder_out.size(1), decoder_out.size(2))

        return decoder_out, batch_state

    def action(self, state, index, output, hidden_now, cell_now, train):
        # print("executing action")
        # train = True
        # if train is True:
        # print(output.size())
        action = []
        if train:
            # print("train")
            for i in range(16):
                if index < len(state.gold[i]):
                    action.append(state.gold[i][index])
                else:
                    action.append("ACT")
            # print(action)
        else:
            # print("eval")
            for i in range(16):
                actionID = self.getMaxindex(self.args, output[i].view(self.args.label_size))
                action.append(self.args.create_alphabet.label_alphabet.from_id(actionID))
            # print(actionID)
            # print(action)
        state.actions.append(action)

        # word_cells = []
        # word_hiddens = []
        # word_cells.append(cell_now)
        # word_hiddens.append(hidden_now)
        # print(action)
        pos_labels = []
        pos_id = []
        # print("length action", len(action))
        for id_batch, act in enumerate(action):
            words = []
            chars = []

            # print(str(id_batch)+"ddddd")
            pos = act.find("#")
            if pos == -1:
                # app
                # if index < len(state.chars[id_batch]):
                #     state.words[id_batch][-1][-1] += (state.chars[id_batch][index])
                # pos_labels.append("#APP")
                # pos_id.append(self.pos_paddingKey)
                if act is not "ACT":
                    pos_id.append(state.pos_id[-1][id_batch])
                else:
                    pos_id.append(self.pos_paddingKey)
                    # state.words[index][-1][-1] += (state.chars[id_batch][index])
            else:
                # if index < len(state.chars[id_batch]):
                #     temp_word = state.chars[id_batch][index]
                #     words.append(temp_word)
                posLabel = act[pos + 1:]
                pos_labels.append(posLabel)
                posId = self.args.create_alphabet.pos_alphabet.loadWord2idAndId2Word(posLabel)
                pos_id.append(posId)
                state.words[id_batch].append(words)
        state.pos_labels.append(pos_labels)
        state.pos_id.append(pos_id)
        state.word_cells.append(cell_now)
        state.word_hiddens.append(hidden_now)


    def getMaxindex_1(self, args, decoder_output):
        # print("get max index ......")
        decoder_output_list = decoder_output.data.tolist()
        print(decoder_output_list)
        maxIndex = decoder_output_list.index(np.max(decoder_output_list))
        return maxIndex

    def getMaxindex(self, args, decoder_output):
        max = decoder_output.data[0]
        maxIndex = 0
        for idx in range(1, args.label_size):
            if decoder_output.data[idx] > max:
                max = decoder_output.data[idx]
                maxIndex = idx
        return maxIndex


    # def getMaxindex(self, decode_out_acc, args):
    #     # print("get max index ......")
    #     max = decode_out_acc.data[0]
    #     maxIndex = 0
    #     for idx in range(1, args.label_size):
    #         if decode_out_acc.data[idx] > max:
    #             max = decode_out_acc.data[idx]
    #             maxIndex = idx
    #     return maxIndex




