import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
import shutil
import random
import hyperparams as hy
import time
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)

"""
    train function
"""

def train(train_iter, dev_iter, test_iter, model_encoder, model_decoder, args):
    # if args.use_cuda:
    #     model_encoder = model_encoder.cuda()
    #     model_decoder = model_decoder.cuda()

    if args.Adam is True:
        print("Adam Training......")
        optimizer_encoder = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model_encoder.parameters()),
                                             lr=args.lr, weight_decay=args.init_weight_decay)
        optimizer_decoder = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model_decoder.parameters()),
                                             lr=args.lr,
                                             weight_decay=args.init_weight_decay)
        # optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)
        # optimizer_decoder = torch.optim.Adam(model_decoder.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)

    steps = 0
    model_count = 0
    # for dropout in train / dev / test
    model_encoder.train()
    model_decoder.train()
    time_list = []
    for epoch in range(1, args.epochs+1):
        print("\n## 第{} 轮迭代，共计迭代 {} 次 ！##\n".format(epoch, args.epochs))
        print("optimizer_encoder now lr is {}".format(optimizer_encoder.param_groups[0].get("lr")))
        print("optimizer_decoder now lr is {} \n".format(optimizer_decoder.param_groups[0].get("lr")))

        # shuffle
        random.shuffle(train_iter)
        # random.shuffle(dev_iter)
        # random.shuffle(test_iter)

        for batch_count, batch_features in enumerate(train_iter):
            # print("batch_count", batch_count)
            # print(batch_features)
            # print(batch_features.inst[batch_count].chars)
            # print(batch_features.batch_length)
            model_encoder.zero_grad()
            model_decoder.zero_grad()

            # print(batch_features.cuda())
            encoder_out = model_encoder(batch_features)
            decoder_out, decoder_out_acc = model_decoder(batch_features, encoder_out)
            # print(decoder_out.size())
            # cal the acc
            train_acc, correct, total_num = cal_train_acc(batch_features, batch_count, decoder_out_acc, args)
            # loss = F.nll_loss(decoder_out, batch_features.gold_features)
            loss = F.cross_entropy(decoder_out, batch_features.gold_features)
            # print("loss {}".format(loss.data[0]))

            steps += 1
            if steps % args.log_interval == 0:
                print("batch_count = {} , loss is {:.6f} , (correct/ total_num) = acc ({} / {}) = {:.6f}%".format(
                    batch_count+1, loss.data[0], correct, total_num, train_acc*100))
            if steps % args.dev_interval == 0:
                eval(dev_iter, model_encoder, model_decoder, args)
            loss.backward()

            optimizer_encoder.step()
            optimizer_decoder.step()




def cal_train_acc(batch_features, batch_count, decode_out_acc, args):
    # print("calculate the acc of train ......")
    correct = 0
    total_num = 0
    for index in range(batch_features.batch_length):
        inst = batch_features.inst[index]
        for char_index in range(inst.chars_size):
            max_index = getMaxindex(decode_out_acc[index][char_index], args)
            if max_index == inst.gold_index[char_index]:
                correct += 1
        total_num += inst.chars_size
    acc = correct / total_num
    return acc, correct, total_num


def cal_pre_fscore(batch_features, decode_out_acc, args):
    # print("calculate the acc of train ......")
    correct_num = 0
    predict_num = 0
    gold_num = 0
    for index in range(batch_features.batch_length):
        inst = batch_features.inst[index]
        for char_index in range(inst.chars_size):
            max_index = getMaxindex(decode_out_acc[index][char_index], args)
            if max_index == inst.gold_index[char_index]:
                correct_num += 1
        gold_num += inst.chars_size

    # return ""

def getMaxindex(decode_out_acc, args):
    # print("get max index ......")
    max = decode_out_acc.data[0]
    maxIndex = 0
    for idx in range(1, args.label_size):
        if decode_out_acc.data[idx] > max:
            max = decode_out_acc.data[idx]
            maxIndex = idx
    return maxIndex


def eval(data_iter, model_encoder, model_decoder, args):
    print("eval function")
    model_encoder.eval()
    model_decoder.eval()
    loss = 0
    for batch_count, batch_features in enumerate(data_iter):
        encoder_out = model_encoder(batch_features)
        decoder_out, decoder_out_acc = model_decoder(batch_features, encoder_out)
        loss = F.cross_entropy(decoder_out, batch_features.gold_features, size_average=False)
        print(loss.data[0])
        cal_pre_fscore(batch_features, decoder_out_acc, args)



