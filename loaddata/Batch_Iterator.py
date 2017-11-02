# coding=utf-8
import torch
from torch.autograd import Variable
import random
from loaddata.common import sep, app, nullkey, paddingkey, unkkey
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)


class Iterators():
    def __init__(self, batch_size=1, examples=None, operator=None):
        self.batch_size = batch_size
        self.example = examples
        self.operator = operator
        self.iterator = []
        self.batch = []
        self.dataIterator = []
        # Iterators.createIterator(self)
        Iterators.convert_word2id(self, self.example, self.operator)

    def convert_word2id(self, insts, operator):
        # print(len(insts))
        for index, inst in enumerate(insts):
            # copy with the word and pos
            for index in range(inst.words_size):
                word = inst.words[index]
                wordID = operator.word_alphabet.loadWord2idAndId2Word(word)
                if wordID is None:
                    wordID = operator.word_UnkkID
                inst.words_index.append(wordID)

                pos = inst.pos[index]
                posID = operator.pos_alphabet.loadWord2idAndId2Word(pos)
                if posID is None:
                    posID = operator.pos_UnkID
                inst.pos_index.append(posID)
            # print(inst.words_index)
            # print(inst.pos_index)
            # copy with the char
            for index in range(inst.chars_size):
                char = inst.chars[index]
                charID = operator.char_alphabet.loadWord2idAndId2Word(char)
                if charID is None:
                    charID = operator.char_UnkID
                inst.chars_index.append(charID)
            # print(inst.chars_index)
            # copy with the bichar_left
            for index in range(inst.bichars_size):
                bichar_left = inst.bichars_left[index]
                bichar_left_ID = operator.bichar_alphabet.loadWord2idAndId2Word(bichar_left)
                if bichar_left_ID is None:
                    bichar_left_ID = operator.bichar_UnkID
                inst.bichars_left_index.append(bichar_left_ID)
            # print(inst.bichars_left_index)

            # copy with the bichar_right
            for index in range(inst.bichars_size):
                bichar_right = inst.bichars_right[index]
                bichar_right_ID = operator.bichar_alphabet.loadWord2idAndId2Word(bichar_right)
                if bichar_right_ID is None:
                    bichar_right_ID = operator.bichar_UnkID
                inst.bichars_right_index.append(bichar_right_ID)
            print(inst.bichars_right_index)

            # copy with the gold
            for index in range(inst.gold_size):
                gold = inst.gold[index]
                goldID = operator.label_alphabet.loadWord2idAndId2Word(gold)
                inst.gold_index.append(goldID)
            # print(inst.gold_index)



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
        for inst in self.example:
            text = []
            label = []
            for word in inst.text:
                # print(word)
                if word in self.word_operator.word2index:
                    text.append(self.word_operator.word2index[word])
                else:
                    text.append(self.unk_id)
            label.append(self.label_operator.word2index[word.label])
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