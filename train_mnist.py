import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import prettytable
import scipy.io as io
import pandas as pd
import argparse
from LeNet_tucker import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).detach().cpu().numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)


def train(m, out_dir):
    iter_loss = []
    train_losses = []
    test_losses = []
    iter_loss_path = os.path.join(out_dir, "iter_loss.csv")
    epoch_loss_path = os.path.join(out_dir, "epoch_loss.csv")

    last_loss = 99999
    mkdirs(os.path.join(out_dir, "models"))
    # optimizer = optim.SGD(m.parameters(), lr=0.003, momentum=0.9)
    optimizer = optim.Adam(m.parameters(), lr=4e-3)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)
    best_test_acc = 0.
    for epoch in range(50):
        train_loss = 0.
        train_acc = 0.
        m.train(mode=True)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = m(data)

            loss = criterion(output, target)
            loss_value = loss.item()
            iter_loss.append(loss_value)
            train_loss += loss_value
            loss.backward()
            optimizer.step()
            acc = compute_accuray(output, target)
            train_acc += acc
        train_losses.append(train_loss / len(train_loader))
        lr_scheduler.step()

        test_loss = 0.
        test_acc = 0.

        m.train(mode=False)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = m(data)

            loss = criterion(output, target)
            loss_value = loss.item()
            iter_loss.append(loss_value)
            test_loss += loss_value
            acc = compute_accuray(output, target)
            test_acc += acc

        test_losses.append(test_loss / len(test_loader))

        print("Epoch {}: train loss is {}, train accuracy is {}; test loss is {}, test accuracy is {}".
                  format(epoch, round(train_loss / len(train_loader), 2),
                         round(train_acc / len(train_loader), 4),
                         round(test_loss / len(test_loader), 2),
                         round(test_acc / len(test_loader), 4)))

        if test_loss / len(test_loader) <= last_loss:
            save_model_path = os.path.join(out_dir, "models", "best_model.tar".format(epoch))
            torch.save({
                "model": m.state_dict(),
                "optimizer": optimizer.state_dict()
            }, save_model_path)
            last_loss = test_loss / len(test_loader)
            best_test_acc = round(test_acc / len(test_loader), 4)

    # # precision and recall
    # m.eval()
    # m.to('cpu')
    # pred_list = torch.tensor([])
    # with torch.no_grad():
    #     for X, y in test_loader:
    #         pred = m(X)
    #         pred_list = torch.cat([pred_list, pred])
    #
    # test_iter1 = torch.utils.data.DataLoader(datasets.MNIST("data", train=False, download=True,
    #                                                         transform=transforms.Compose([transforms.ToTensor(), ])),
    #                                          batch_size=10000, shuffle=False, num_workers=2)
    # features, labels = next(iter(test_iter1))
    # print(labels.shape)
    #
    # train_result = np.zeros((10, 10), dtype=int)
    # for i in range(10000):
    #     train_result[labels[i]][np.argmax(pred_list[i])] += 1
    # result_table = prettytable.PrettyTable()
    # result_table.field_names = ['Type', 'Accuracy(精确率)', 'Recall(召回率)', 'F1_Score']
    # class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    # for i in range(10):
    #     precision = train_result[i][i] / train_result.sum(axis=0)[i]
    #     recall = train_result[i][i] / train_result.sum(axis=1)[i]
    #     result_table.add_row([class_names[i], np.round(precision, 3), np.round(recall, 3),
    #                           np.round(precision * recall * 2 / (precision + recall), 3)])
    # print(result_table)

    df = pd.DataFrame()
    df["iteration"] = np.arange(0, len(iter_loss))
    df["loss"] = iter_loss
    df.to_csv(iter_loss_path, index=False)

    df = pd.DataFrame()
    df["epoch"] = np.arange(0, 50)
    df["train_loss"] = train_losses
    df["test_loss"] = test_losses
    df.to_csv(epoch_loss_path, index=False)

    print('Finish training!')
    print('The best test accuracy is:', best_test_acc)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--com_rate', '--compression-rate', default=0.5, type=float)
    parser.add_argument('--ker_size', '--kernel-size', default=3, type=int)
    parser.add_argument('--com_ker_size', '--compression-kernel-size', default=3, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])), batch_size=128, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss().cuda()

    start = time.time()

    net = TuckerLeNet(ch_com_rate=args.com_rate, kernel_size=args.ker_size, compress_size=args.com_ker_size).to(device)
    # net = LeNet().to(device)
    net.train(mode=True)
    train(net, "lenet")

    elapsed = (time.time() - start)
    print('channel compression rate:', args.com_rate)
    print('kernel size:', args.ker_size)
    print('compression kernel size:', args.com_ker_size)
    print("Finished. Time used:", elapsed)

