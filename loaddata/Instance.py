# coding=utf-8

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
        self.bichars_size = 0

        self.gold = []
        self.pos = []
        self.gold_pos = []
        self.gold_seg = []
        self.gold_size = 0

        self.words_index = []
        self.chars_index = []
        self.bichars_left_index = []
        self.bichars_right_index = []
        self.pos_index = []
        self.gold_index = []


class Batch_Features:
    def __init__(self):
        self.batch_length = 0
        self.word_features = None
        self.pos_features = None
        self.char_features = None
        self.bichar_left_features = None
        self.bichar_right_features = None
        self.gold_features = None

    def cuda(self):
        self.word_features.cuda()
        self.pos_features.cuda()
        self.char_features.cuda()
        self.bichar_left_features.cuda()
        self.bichar_right_features.cuda()
        self.gold_features.cuda()
