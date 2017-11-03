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
    if args.use_cuda:
        model_encoder = model_encoder.cuda()
        model_decoder = model_decoder.cuda()

    if args.Adam is True:
        print("Adam Training......")
        optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)
        optimizer_decoder = torch.optim.Adam(model_decoder.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)

    steps = 0
    model_count = 0
    # for dropout in train / dev / test
    model_encoder.train()
    model_decoder.train()
    time_list = []
    for epoch in range(1, args.epochs+1):
        print("\n## 第{} 轮迭代，共计迭代 {} 次 ！##\n".format(epoch, args.epochs))
        print("optimizer_encoder now lr is {} \n".format(optimizer_encoder.param_groups[0].get("lr")))
        print("optimizer_decoder now lr is {} \n".format(optimizer_decoder.param_groups[0].get("lr")))
        for batch_features in train_iter:
            print(batch_features.gold_features)
            print(batch_features.char_features)














def eval(data_iter, model, args, scheduler):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        target.data.sub_(1)
        # feature, target = batch.text, batch.label.data.sub_(1)
        if args.cuda is True:
            feature, target = feature.cuda(), target.cuda()
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        # feature.data.t_(),\
        # target.data.sub_(1)  # batch first, index align
        # target = autograd.Variable(target)

        model.hidden = model.init_hidden(args.lstm_num_layers, args.batch_size)
        if feature.size(1) != args.batch_size:
            # print("aaa")
            # continue
            model.hidden = model.init_hidden(args.lstm_num_layers, feature.size(1))
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        # scheduler.step(loss.data[0])

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = float(corrects)/size * 100.0
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))


def test_eval(data_iter, model, save_path, args, model_count):
    # print(save_path)
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        target.data.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        # feature.data.t_()
        # target.data.sub_(1)  # batch first, index align
        # target = autograd.Variable(target)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        model.hidden = model.init_hidden(args.lstm_num_layers, args.batch_size)
        if feature.size(1) != args.batch_size:
            # continue
            model.hidden = model.init_hidden(args.lstm_num_layers, feature.size(1))
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = float(corrects)/size * 100.0
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    print("model_count {}".format(model_count))
    # test result
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt", "a")
    else:
        file = open("./Test_Result.txt", "w")
    file.write("model " + save_path + "\n")
    file.write("Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n".format(avg_loss, accuracy, corrects, size))
    file.write("model_count {} \n".format(model_count))
    file.write("\n")
    file.close()
    # calculate the best score in current file
    resultlist = []
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt")
        for line in file.readlines():
            if line[:10] == "Evaluation":
                resultlist.append(float(line[34:41]))
        result = sorted(resultlist)
        file.close()
        file = open("./Test_Result.txt", "a")
        file.write("\nThe Current Best Result is : " + str(result[len(result) - 1]))
        file.write("\n\n")
        file.close()
    shutil.copy("./Test_Result.txt", "./snapshot/" + args.mulu + "/Test_Result.txt")
    # whether to delete the model after test acc so that to save space
    if os.path.isfile(save_path) and args.rm_model is True:
        os.remove(save_path)
