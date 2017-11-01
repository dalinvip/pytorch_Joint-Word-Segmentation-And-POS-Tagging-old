# coding=utf-8

import os
import re
import torch
import shutil
import random
# fix the random seed
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)


# common later
unkkey = '-unk-'
nullkey = '-NULL-'
paddingkey = '-padding-'
sep = 'SEP'
app = 'APP'


"""
   init instance
"""
class instance():

    def __init__(self):
        # print("init instance......")

        self.words = []
        self.words_size = 0
        self.chars = []
        self.chars_size = 0
        self.bichars_left = []
        self.bichars_right = []
        self.bichars_size = []

        self.gold = []
        self.pos = []
        self.gold_pos = []
        self.gold_seg = []
        self.gold_size = 0


class load_data():

    def __init__(self):
        print("load data for train/dev/test")

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def loaddate(self, path, shuffle=False):
        print("loading {} data......".format(path))
        assert path is not None
        insts = []
        with open(path, encoding="UTF-8") as f:
            lines = f.readlines()
            for line in lines:
                # init instance
                inst = instance()
                line = line.split(" ")
                count = 0
                for word_pos in line:
                    # segment the word and pos in line
                    word, _, label = word_pos.partition("_")
                    word_length = len(word)
                    inst.words.append(word)
                    inst.gold_seg.append("[" + str(count) + "," + str(count + word_length) + "]")
                    inst.gold_pos.append("[" + str(count) + "," + str(count + word_length) + "]" + label)
                    count += word_length
                    for i in range(word_length):
                        char = word[i]
                        # print(char)
                        inst.chars.append(char)
                        if i == 0:
                            inst.gold.append(sep + "#" + label)
                            inst.pos.append(label)
                        else:
                            inst.gold.append(app)
                char_number = len(inst.chars)
                for i in range(char_number):
                    # copy with the left bichars
                    if i is 0:
                        inst.bichars_left.append(nullkey + inst.chars[i])
                    else:
                        inst.bichars_left.append(inst.chars[i - 1] + inst.chars[i])
                    # copy with the right bichars
                    if i == char_number - 1:
                        inst.bichars_right.append(inst.chars[i] + nullkey)
                    else:
                        inst.bichars_right.append(inst.chars[i] + inst.chars[i + 1])
                # char/word size
                inst.chars_size = len(inst.chars)
                inst.words_size = len(inst.words)
                inst.bichars_size = len(inst.bichars_left)
                inst.gold_size = len(inst.gold)
                # add one inst that represent one sentence into the list
                insts.append(inst)
        if shuffle is True:
            print("shuffle tha data......")
            random.shuffle(insts)
        # return all sentence in data
        # print(insts[-1].words)
        return insts


if __name__ == "__main__":
    print("Test dataloader........")
    load_data = load_data()
    load_data.loaddate("../pos_test_data/train.ctb60.pos.hwc", shuffle=True)
    # load_data.loaddate("../pos_test_data/test.ctb60.pos.hwc")
