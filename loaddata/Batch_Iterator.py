# coding=utf-8
import torch
from torch.autograd import Variable
import random
from loaddata.common import sep, app, nullkey, paddingkey, unkkey
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)


class Iterators():
    def __init__(self, batch_size=1, examples=None, word_operator=None,
                 label_operator=None, unk_id=None, pad_id=None):
        self.batch_size = batch_size
        self.example = examples
        self.iterator = []
        self.batch = []
        self.word_operator = word_operator
        self.label_operator = label_operator
        self.unk_id = unk_id
        self.pad_id = pad_id
        self.dataIterator = []
        Iterators.createIterator(self)

    def batch(self, datasets, batch_size, max):
        for data in datasets:
            if len(data[0]) == max:
                continue
            for i in range(max - len(data[0])):
                # data[0].append(self.unk_id)
                data[0].append(self.pad_id)
        minibatch = []
        minibatch_text = []
        minibatch_label = []
        for ex in datasets:
            minibatch_text.append(ex[0])
            minibatch_label.append(ex[1][0])
            if len(minibatch_text) == batch_size:
                minibatch_text = Variable(torch.LongTensor(minibatch_text))
                minibatch_label = Variable(torch.LongTensor(minibatch_label))
                minibatch.append(minibatch_text)
                minibatch.append(minibatch_label)
                return minibatch
        if minibatch_text or minibatch_label:
            minibatch_text = Variable(torch.LongTensor(minibatch_text))
            minibatch_label = Variable(torch.LongTensor(minibatch_label))
            minibatch.append(minibatch_text)
            minibatch.append(minibatch_label)
            return minibatch

    def createIterator(self):
        # print("aaa")
        batch = []
        count = 0
        max = 0
        for word_label in self.example:
            text = []
            label = []
            for word in word_label.text:
                # print(word)
                if word in self.word_operator.word2index:
                    text.append(self.word_operator.word2index[word])
                else:
                    text.append(self.unk_id)
            label.append(self.label_operator.word2index[word_label.label])
            batch.append((text, label))
            if len(batch[-1][0]) > max:
                max = len(batch[-1][0])
            count += 1
            # print(batch)
            if len(batch) == self.batch_size or count == len(self.example):
                # print("aaaa")
                batchs = Iterators.batch(self, datasets=batch, batch_size=self.batch_size, max=max)
                self.dataIterator.append(batchs)
                batch = []
                max = 0