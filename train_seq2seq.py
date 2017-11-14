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

            loss.backward()

            if args.init_clip_max_norm is not None:
                utils.clip_grad_norm(model_encoder.parameters(), max_norm=args.init_clip_max_norm)
                utils.clip_grad_norm(model_decoder.parameters(), max_norm=args.init_clip_max_norm)

            optimizer_encoder.step()
            optimizer_decoder.step()

            steps += 1
            if steps % args.log_interval == 0:
                # print("batch_count = {} , loss is {:.6f} , (correct/ total_num) = acc ({} / {}) = {:.6f}%\r".format(
                #     batch_count+1, loss.data[0], correct, total_num, train_acc*100))
                sys.stdout.write("\rbatch_count = [{}] , loss is {:.6f} , (correct/ total_num) = acc ({} / {}) = {:.6f}%".format(
                    batch_count+1, loss.data[0], correct, total_num, train_acc*100))
            if steps % args.dev_interval == 0:
                print("\ndev F-score")
                dev_eval_pos.clear()
                dev_eval_seg.clear()
                eval(dev_iter, model_encoder, model_decoder, args, dev_eval_seg, dev_eval_pos)
                model_encoder.train()
                model_decoder.train()
            if steps % args.test_interval == 0:
                print("test F-score")
                test_eval_pos.clear()
                test_eval_seg.clear()
                eval(test_iter, model_encoder, model_decoder, args, test_eval_seg, test_eval_pos)
                print("\n")
                model_encoder.train()
                model_decoder.train()


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


def cal_pre_fscore(batch_features, decode_out_acc, args, eval_seg, eval_pos):
    # print("calculate the acc of train ......")
    correct = 0
    predict = 0
    gold = 0
    for index in range(batch_features.batch_length):
        inst = batch_features.inst[index]
        label_list = []
        # print("aaa", label_list)
        for char_index in range(inst.chars_size):
            max_index = getMaxindex(decode_out_acc[index][char_index], args)
            label = args.create_alphabet.label_alphabet.from_id(max_index)
            # print(label)
            if sep in label or len(label_list) > 0:
                label_list.append(label)
        # print(label_list)
        jointPRF(label_list, inst, eval_seg, eval_pos)
        # jointPRF_m(label_list, inst, eval_seg, eval_pos)


def jointPRF_m(label_list, inst, seg_eval, pos_eval):
    # words = state.words
    # posLabels = state.pos_labels
    count = 0
    predict_seg = []
    predict_pos = []
    sep_list = []
    pos_list = []
    # print(label_list)
    for index, value in enumerate(label_list):
        # if sep not in label_list[index]:
        #     continue
        label_sep, _, label_pos = value.partition("#")
        sep_list.append(label_sep)
        pos_list.append(label_pos)

    print("**/*************/**************************************************")
    print(sep_list)
    print(pos_list)
    word_list = {}
    word_number = 0
    for index, value in enumerate(sep_list):
        if sep_list[0] == sep:
            if value == sep or sep_list[0] == app:
                word_number += 1
                word_list[word_number] = 1
            else:
                # print(word_number)
                word_list[word_number] += 1
        if sep_list[0] == app:
            if value == app and index == 0:
                word_number = 1
                word_list[word_number] = 1
            else:
                if value == app or sep_list[0] == sep:
                    word_list[word_number] += 1
                else:
                    word_number += 1
                    word_list[word_number] = 1
                # print(word_number)

    # print(word_list)

    count = 0
    for index in word_list:
        predict_seg.append('[' + str(count) + ',' + str(count + word_list[index]) + ']')
        predict_pos.append('[' + str(count) + ',' + str(count + word_list[index]) + ']' + pos_list[count])
        count += word_list[index]
    # print(predict_pos)
    # print(predict_seg)
    # print("\n\n\n ****************")

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


def jointPRF(label_list, inst, seg_eval, pos_eval):
    # words = state.words
    # posLabels = state.pos_labels
    count = 0
    predict_seg = []
    predict_pos = []
    sep_list = []
    pos_list = []
    # print(label_list)
    for index, value in enumerate(label_list):
        # if sep not in label_list[index]:
        #     continue
        label_sep, _, label_pos = value.partition("#")
        sep_list.append(label_sep)
        pos_list.append(label_pos)

    # print("**/*************/**************************************************")
    # print(sep_list)
    # print(pos_list)
    word_list = {}
    word_number = 0
    for index, value in enumerate(sep_list):
        if sep_list[0] == sep:
            if value == sep or sep_list[0] == app:
                word_number += 1
                word_list[word_number] = 1
            else:
                # print(word_number)
                word_list[word_number] += 1
        if sep_list[0] == app:
            if value == app and index == 0:
                word_number = 1
                word_list[word_number] = 1
            else:
                if value == app or sep_list[0] == sep:
                    word_list[word_number] += 1
                else:
                    word_number += 1
                    word_list[word_number] = 1
                # print(word_number)

    # print(word_list)

    count = 0
    for index in word_list:
        predict_seg.append('[' + str(count) + ',' + str(count + word_list[index]) + ']')
        predict_pos.append('[' + str(count) + ',' + str(count + word_list[index]) + ']' + pos_list[count])
        count += word_list[index]
    # print(predict_pos)
    # print(predict_seg)
    # print("\n\n\n ****************")

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
    model_encoder.eval()
    model_decoder.eval()
    loss = 0
    # eval_seg = Eval()
    # eval_pos = Eval()
    for batch_count, batch_features in enumerate(data_iter):
        encoder_out = model_encoder(batch_features)
        decoder_out, decoder_out_acc = model_decoder(batch_features, encoder_out)
        # loss = F.cross_entropy(decoder_out, batch_features.gold_features, size_average=False)
        # print(loss.data[0])
        cal_pre_fscore(batch_features, decoder_out_acc, args, eval_seg, eval_pos)

    p, r, f = eval_seg.getFscore()
    # print("\n")
    print("seg dev: precision = {}%  recall = {}% , f-score = {}%".format(p, r, f))
    p, r, f = eval_pos.getFscore()
    print("pos dev: precision = {}%  recall = {}% , f-score = {}%".format(p, r, f))
    # print("\n")



