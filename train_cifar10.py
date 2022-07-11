import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import prettytable

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import ResNet
import ResNet_tucker
import vgg
import vgg_tucker
import CMT_tucker

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_names_res = sorted(name for name in ResNet.__dict__
                         if name.islower() and not name.startswith("__")
                         and name.startswith("resnet")
                         and callable(ResNet.__dict__[name]))

model_names_v = sorted(name for name in vgg.__dict__
                       if name.islower() and name.startswith("vgg")
                       and callable(vgg.__dict__[name]))

model_names_tck = sorted(name for name in ResNet_tucker.__dict__
                         if not name.startswith("__")
                         and name.startswith("Tuckerresnet")
                         and callable(ResNet_tucker.__dict__[name]))

model_names_vgg = sorted(name for name in vgg_tucker.__dict__
                         if not name.startswith("__")
                         and name.startswith("Tuckervgg")
                         and callable(vgg_tucker.__dict__[name]))

CMT = ['CMT_Ti', 'CMT_XS', 'CMT_S', 'CMT_B']


model_names = model_names_res + model_names_v + model_names_tck + model_names_vgg + CMT

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')  # resnet 200  vgg 300
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')  # resnet 0.1  vgg 0.05
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')  # resnet 1e-4  vgg 5e-4
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--com_rate', '--compression-rate', default=0.5, type=float)
parser.add_argument('--ker_size', '--kernel-size', default=3, type=int)
parser.add_argument('--com_ker_size', '--compression-kernel-size', default=3, type=int)
parser.add_argument('--pre_path', default=None, type=str)

best_prec1 = 0
best_prec5 = 0


def main():
    begin_time = time.clock()
    global args, best_prec1, best_prec5
    args = parser.parse_args()

    args.save_dir = args.arch
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.arch in model_names_res:
        model = ResNet.__dict__[args.arch]()
    if args.arch in model_names_tck:
        model = ResNet_tucker.__dict__[args.arch](ch_com_rate=args.com_rate, kernel_size=args.ker_size,
                                                  compress_size=args.com_ker_size)
    if args.arch in model_names_v:
        model = vgg.__dict__[args.arch]()
        args.epochs = 300
        args.lr = 0.05
        args.weight_decay = 5e-4

    if args.arch in model_names_vgg:
        model = vgg_tucker.__dict__[args.arch](ch_com_rate=args.com_rate, kernel_size=args.ker_size,
                                               compress_size=args.com_ker_size)
        args.epochs = 300
        args.lr = 0.05
        # args.weight_decay = 5e-4

    if args.arch in CMT:
        model = CMT_tucker.__dict__[args.arch](ch_com_rate=args.com_rate, kernel_size=args.ker_size,
                                               compress_size=args.com_ker_size, num_class=10)
        args.batch_size = 64
        args.epochs = 150
        args.lr = 6e-5
        args.weight_decay = 1e-5

    model.cuda()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, ':', param.size())

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.arch in CMT:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    if args.resume:
        optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': 5e-4}], 5e-4,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.arch in model_names_tck or args.arch in model_names_res:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[100, 200], last_epoch=args.start_epoch - 1)
    if args.arch in model_names_vgg or args.arch in model_names_v:
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.5,
        #                                                     milestones=[30, 60, 90, 120, 150, 180, 210, 240, 270],
        #                                                     last_epoch=args.start_epoch - 1)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[100, 200], last_epoch=args.start_epoch - 1)

    if args.arch in CMT:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.arch in ['resnet1202', 'resnet110', 'Tuckerresnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_prec5 = max(prec5, best_prec5)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best,
                filename=os.path.join(args.save_dir, 'checkpoint_' + str(args.com_rate) + '_' + str(args.com_ker_size)
                                      + '_' + str(args.ker_size) + '.th'))

        if is_best:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'model' + str(args.com_rate) + '_' + str(args.com_ker_size)
                                              + '_' + str(args.ker_size) + '.th'))

    # # precision and recall
    # model.eval()
    # model.to('cpu')
    # pred_list = torch.tensor([])
    # with torch.no_grad():
    #     for X, y in val_loader:
    #         pred = model(X)
    #         pred_list = torch.cat([pred_list, pred])
    #
    # test_iter1 = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
    #             transforms.Resize(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    #         ])),
    #                                          batch_size=10000, shuffle=False, num_workers=2)
    # features, labels = next(iter(test_iter1))
    # print(labels.shape)
    #
    # train_result = np.zeros((10, 10), dtype=int)
    # for i in range(10000):
    #     train_result[labels[i]][np.argmax(pred_list[i])] += 1
    # result_table = prettytable.PrettyTable()
    # result_table.field_names = ['Type', 'Accuracy(精确率)', 'Recall(召回率)', 'F1_Score']
    # class_names = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
    # for i in range(10):
    #     precision = train_result[i][i] / train_result.sum(axis=0)[i]
    #     recall = train_result[i][i] / train_result.sum(axis=1)[i]
    #     result_table.add_row([class_names[i], np.round(precision, 3), np.round(recall, 3),
    #                           np.round(precision * recall * 2 / (precision + recall), 3)])
    # print(result_table)

    print('Finish training!')
    print('best Prec@1 is:', best_prec1)
    print('best Prec@5 is:', best_prec5)
    end_time = time.clock()
    print(end_time - begin_time)
    print('network:', args.arch)
    print('channel compression rate:', args.com_rate)
    print('kernel size:', args.ker_size)
    print('compression kernel size:', args.com_ker_size)


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f}' '* Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
