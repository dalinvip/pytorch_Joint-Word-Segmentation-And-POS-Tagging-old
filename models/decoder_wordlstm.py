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

        # self.lstm = nn.LSTM(input_size=self.args.hidden_size, hidden_size=self.args.rnn_hidden_dim, bias=True)
        self.lstmcell = nn.LSTMCell(input_size=self.args.hidden_size, hidden_size=self.args.rnn_hidden_dim, bias=True)
        init.xavier_uniform(self.lstmcell.weight_ih)
        init.xavier_uniform(self.lstmcell.weight_hh)
        self.lstmcell.bias_hh.data.uniform_(-np.sqrt(6 / (self.args.rnn_hidden_dim + 1)),
                                            np.sqrt(6 / (self.args.rnn_hidden_dim + 1)))
        self.lstmcell.bias_ih.data.uniform_(-np.sqrt(6 / (self.args.rnn_hidden_dim + 1)),
                                            np.sqrt(6 / (self.args.rnn_hidden_dim + 1)))

        self.pos_embed = nn.Embedding(num_embeddings=self.args.pos_size, embedding_dim=self.args.pos_dim)
        init.uniform(self.pos_embed.weight,
                     a=-np.sqrt(3 / self.args.pos_dim),
                     b=np.sqrt(3 / self.args.pos_dim))
        self.pos_embed.weight.requires_grad = True

        self.linear = nn.Linear(in_features=self.args.rnn_hidden_dim * 2 + self.args.hidden_size,
                                out_features=self.args.label_size, bias=False)

        # self.non_linear = nn.Linear(in_features=self.args.rnn_hidden_dim * 2, out_features=self.args.hidden_size,
        #                             bias=True)

        self.combine_linear = nn.Linear(in_features=self.args.rnn_hidden_dim * 2 + self.args.pos_dim,
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

        self.z_bucket = Variable(torch.zeros(1, self.args.hidden_size))
        self.h_bucket = Variable(torch.zeros(1, self.args.rnn_hidden_dim))
        self.c_bucket = Variable(torch.zeros(1, self.args.rnn_hidden_dim))
        if self.args.use_cuda is True:
            self.z_bucket = self.z_bucket.cuda()
            self.h_bucket = self.h_bucket.cuda()
            self.c_bucket = self.c_bucket.cuda()

    def forward(self, features, encoder_out, train=False):
        # print(encoder_out.size())
        # print("Decoder forward")
        batch_length = features.batch_length
        # char_features_num = features.char_features.size(1)
        char_features_num = encoder_out.size(1)
        batch_output = []
        batch_state = []
        for id_batch in range(batch_length):
            feature = features.inst[id_batch]
            state = state_instance(feature)
            sent_output = []
            real_char_num = feature.chars_size
            for id_char in range(char_features_num):
                if id_char < real_char_num:
                    hidden_now, cell_now = self.word_lstm(state, id_char, encoder_out[id_batch])
                    # print(hidden_now)

                    # not use lstm
                    # v = torch.cat((self.bucket_rnn, encoder_out[id_batch][id_char].view(1, self.args.rnn_hidden_dim * 2)), 1)
                    # use lstm
                    v = torch.cat((hidden_now, encoder_out[id_batch][id_char].view(1, self.args.rnn_hidden_dim * 2)), 1)
                    # print("232", v.size())
                    output = self.linear(v)
                    if id_char is 0:
                        # print("oooooo")
                        output.data[0][self.args.create_alphabet.appID] = -10e+99
                    # self.action(state, id_char, encoder_out[id_batch], output, train)
                    self.action(state, id_char, output, hidden_now, cell_now, train)
                    sent_output.append(output)
                else:
                    sent_output.append(self.bucket)
            sent_output = torch.cat(sent_output, 0)
            batch_output.append(sent_output)
            batch_state.append(state)

        batch_output = torch.cat(batch_output, 0)
        batch_output = self.softmax(batch_output)
        # print("batch_output", batch_output.size())
        decoder_out_acc = batch_output.view(batch_length, encoder_out.size(1), -1)
        # print("de", decoder_out_acc.size())
        return batch_output, batch_state, decoder_out_acc

    def word_lstm(self, state, index, encoder_out):
        # print("executing word lstm")
        if index is 0:
            # print("index is zero")
            hidden_last = self.h_bucket
            cell_last = self.c_bucket
            z = self.z_bucket
        else:
            # print("index is not zero")
            hidden_last = state.word_hiddens[-1]
            cell_last = state.word_cells[-1]
            if len(state.pos_id) > 0:
                last_pos = Variable(torch.zeros(1)).type(torch.LongTensor)
                if self.args.use_cuda is True:
                    last_pos = last_pos.cuda()
                last_pos.data[0] = state.pos_id[-1]
                # print(last_pos)
                last_pos_embed = self.dropout(self.pos_embed(last_pos))
            if len(state.words) > 0:
                last_word_len = len(state.words[-1])
                start = index - last_word_len
                end = index
                chars_embed = []
                for i in range(start, end):
                    chars_embed.append(encoder_out[i].view(1, 1, 2 * self.args.rnn_hidden_dim))
                chars_embed = torch.cat(chars_embed, 1)
                last_word_embed = F.avg_pool1d(chars_embed.permute(0, 2, 1), last_word_len).view(1, self.args.rnn_hidden_dim * 2)

            concat = torch.cat((last_pos_embed, last_word_embed), 1)
            z = self.dropout(F.tanh(self.combine_linear(concat)))

        # print("z", z.size())
        # print("hidden", hidden_last.size())
        # print("cell", cell_last.size())

        hidden_now, cell_now = self.lstmcell(z, (hidden_last, cell_last))
        state.all_h.append(hidden_now)
        state.all_c.append(cell_now)

        return hidden_now, cell_now

    def action(self, state, index, output, hidden_now, cell_now, train):
        # print("executing action")
        # train = True
        if train is True:
            action = state.gold[index]
        else:
            actionID =self.getMaxindex(output.view(self.args.label_size), self.args)
            action = self.args.create_alphabet.label_alphabet.from_id(actionID)
            # print(actionID)
            # print(action)
        state.actions.append(action)

        pos = action.find("#")
        if pos is -1:
            # app
            state.words[-1] += state.chars[index]
        else:
            temp_word = state.chars[index]
            state.words.append(temp_word)
            posLabel = action[pos + 1:]
            state.pos_labels.append(posLabel)
            posId = self.args.create_alphabet.pos_alphabet.loadWord2idAndId2Word(posLabel)
            state.pos_id.append(posId)
            state.word_cells.append(cell_now)
            state.word_hiddens.append(hidden_now)

    def getMaxindex(self, decode_out_acc, args):
        # print("get max index ......")
        max = decode_out_acc.data[0]
        maxIndex = 0
        for idx in range(1, args.label_size):
            if decode_out_acc.data[idx] > max:
                max = decode_out_acc.data[idx]
                maxIndex = idx
        return maxIndex




