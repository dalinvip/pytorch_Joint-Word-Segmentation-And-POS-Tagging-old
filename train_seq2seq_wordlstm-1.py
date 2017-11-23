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
from eval import Eval
from loaddata.common import sep, app, paddingkey, unkkey, nullkey
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
        model_encoder_parameters = filter(lambda p: p.requires_grad, model_encoder.parameters())
        model_decoder_parameters = filter(lambda p: p.requires_grad, model_decoder.parameters())
        optimizer_encoder = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model_encoder.parameters()),
                                             lr=args.lr,
                                             weight_decay=args.init_weight_decay)
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
    dev_eval_seg = Eval()
    dev_eval_pos = Eval()
    test_eval_seg = Eval()
    test_eval_pos = Eval()
    for epoch in range(1, args.epochs+1):
        print("\n## 第{} 轮迭代，共计迭代 {} 次 ！##\n".format(epoch, args.epochs))
        print("optimizer_encoder now lr is {}".format(optimizer_encoder.param_groups[0].get("lr")))
        print("optimizer_decoder now lr is {} \n".format(optimizer_decoder.param_groups[0].get("lr")))

        # shuffle
        random.shuffle(train_iter)
        # random.shuffle(dev_iter)
        # random.shuffle(test_iter)

        model_encoder.train()
        model_decoder.train()

        for batch_count, batch_features in enumerate(train_iter):
            # print("batch_count", batch_count)
            # print(batch_features)
            # print(batch_features.inst[batch_count].chars)
            # print(batch_features.batch_length)
            model_encoder.zero_grad()
            model_decoder.zero_grad()

            # print(batch_features.cuda())
            encoder_out = model_encoder(batch_features)
            decoder_out, state, decoder_out_acc = model_decoder(batch_features, encoder_out, train=True)
            # print(decoder_out.size())
            # cal the acc
            # decoder_out_acc =
            train_acc, correct, total_num = cal_train_acc(batch_features, batch_count, decoder_out_acc, args)
            # loss = F.nll_loss(decoder_out, batch_features.gold_features)
            # loss = F.cross_entropy(decoder_out, batch_features.gold_features)
            loss = torch.nn.functional.nll_loss(decoder_out, batch_features.gold_features)
            # print("loss {}".format(loss.data[0]))

            loss.backward()

            if args.init_clip_max_norm is not None:
                # utils.clip_grad_norm(model_encoder.parameters(), max_norm=args.init_clip_max_norm)
                # utils.clip_grad_norm(model_decoder.parameters(), max_norm=args.init_clip_max_norm)
                utils.clip_grad_norm(model_encoder_parameters, max_norm=args.init_clip_max_norm)
                utils.clip_grad_norm(model_decoder_parameters, max_norm=args.init_clip_max_norm)

            optimizer_encoder.step()
            optimizer_decoder.step()

            steps += 1
            if steps % args.log_interval == 0:
                # print("batch_count = {} , loss is {:.6f} , (correct/ total_num) = acc ({} / {}) = {:.6f}%\r".format(
                #     batch_count+1, loss.data[0], correct, total_num, train_acc*100))
                sys.stdout.write("\rbatch_count = [{}] , loss is {:.6f} , (correct/ total_num) = acc ({} / {}) = "
                                 "{:.6f}%".format(batch_count+1, loss.data[0], correct, total_num, train_acc*100))
            if steps % args.dev_interval == 0:
                print("\ndev F-score")
                dev_eval_pos.clear()
                dev_eval_seg.clear()
                eval(dev_iter, model_encoder, model_decoder, args, dev_eval_seg, dev_eval_pos)
                # model_encoder.train()
                # model_decoder.train()
            if steps % args.test_interval == 0:
                print("test F-score")
                test_eval_pos.clear()
                test_eval_seg.clear()
                eval(test_iter, model_encoder, model_decoder, args, test_eval_seg, test_eval_pos)
                print("\n")
        model_encoder.eval()
        model_decoder.eval()
        if steps is not 0:
            print("\none epoch dev F-score")
            dev_eval_pos.clear()
            dev_eval_seg.clear()
            eval(dev_iter, model_encoder, model_decoder, args, dev_eval_seg, dev_eval_pos)
            # model_encoder.train()
            # model_decoder.train()
        if steps is not 0:
            print("one epoch test F-score")
            test_eval_pos.clear()
            test_eval_seg.clear()
            eval(test_iter, model_encoder, model_decoder, args, test_eval_seg, test_eval_pos)
            print("\n")


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


def jointPRF(inst, state, seg_eval, pos_eval):
    words = state.words
    posLabels = state.pos_labels
    count = 0
    predict_seg = []
    predict_pos = []

    for idx in range(len(words)):
        w = words[idx]
        posLabel = posLabels[idx]
        predict_seg.append('[' + str(count) + ',' + str(count + len(w)) + ']')
        predict_pos.append('[' + str(count) + ',' + str(count + len(w)) + ']' + posLabel)
        count += len(w)

    seg_eval.gold_num += len(inst.gold_seg)
    seg_eval.predict_num += len(predict_seg)
    for p in predict_seg:
        if p in inst.gold_seg:
            seg_eval.correct_num += 1

    pos_eval.gold_num += len(inst.gold_pos)
    pos_eval.predict_num += len(predict_pos)
    for p in predict_pos:
        if p in inst.gold_pos:
            pos_eval.correct_num += 1


def getMaxindex(decode_out_acc, args):
    # print("get max index ......")
    max = decode_out_acc.data[0]
    maxIndex = 0
    for idx in range(1, args.label_size):
        if decode_out_acc.data[idx] > max:
            max = decode_out_acc.data[idx]
            maxIndex = idx
    return maxIndex


def eval(data_iter, model_encoder, model_decoder, args, eval_seg, eval_pos):
    # print("eval function")
    # model_encoder.eval()
    # model_decoder.eval()
    loss = 0
    # eval_seg = Eval()
    # eval_pos = Eval()
    for batch_features in data_iter:
        encoder_out = model_encoder(batch_features)
        decoder_out, state, decoder_out_acc = model_decoder(batch_features, encoder_out, train=False)
        for i in range(batch_features.batch_length):
            jointPRF(batch_features.inst[i], state[i], eval_seg, eval_pos)
        # batch_features.inst
        # decoder_out, decoder_out_acc = model_decoder(batch_features, encoder_out)
        # loss = F.cross_entropy(decoder_out, batch_features.gold_features, size_average=False)
        # print(loss.data[0])
        # cal_pre_fscore(batch_features, decoder_out_acc, args, eval_seg, eval_pos)

    p, r, f = eval_seg.getFscore()
    # print("\n")
    print("seg dev: precision = {}%  recall = {}% , f-score = {}%".format(p, r, f))
    p, r, f = eval_pos.getFscore()
    print("pos dev: precision = {}%  recall = {}% , f-score = {}%".format(p, r, f))
    # print("\n")

    # model_encoder.train()
    # model_decoder.train()



