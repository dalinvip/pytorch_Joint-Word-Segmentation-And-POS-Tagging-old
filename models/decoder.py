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
    sequence to sequence Decoder model
"""


class Decoder(nn.Module):

    def __init__(self, args):
        print("Decoder model")
        super(Decoder, self).__init__()
        self.args = args

        self.liner = nn.Linear(in_features=200, out_features=2)
    def forward(self, features):
        print("Decoder forward")

        return ""


