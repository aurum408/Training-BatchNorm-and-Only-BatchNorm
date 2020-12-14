import os
import shutil
import time
from datetime import date

import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
from resnet import freeze_model, unfreeze_model, test
from lbfgsnew import LBFGSNew
from config import parser


lambda1=0.000001
lambda2=0.001

args = parser.parse_args()
global best_prec1

def main():
    best_prec1 = 0


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    trainable = args.trainable
    if trainable == "none":
        trainable = "freeze"

    if args.use_lbfgs:
        opt = "LBFGS"
    else:
        opt = "SGD"

    if args.wide_resnet:
        wide = "wide"
    else:
        wide = ""

    datetime = date.today().strftime("%b-%d-%Y")

    exp_name = "{}{}{}{}_{}".format(args.arch, wide, opt, trainable, datetime)
    logfile = open(os.path.join(args.save_dir, "{}.txt".format(exp_name)), "a")

    if args.wide_resnet:
        # use wide residual net https://arxiv.org/abs/1605.07146
        model = torchvision.models.resnet.wide_resnet50_2()
    else:
        model = resnet.__dict__[args.arch]()
    if trainable == "freeze":
        freeze_model(model)
    elif trainable == "bn":
        freeze_model(model)
        unfreeze_model(model, ["gamma", "beta"])

    print(test(model), file=logfile)

    model.cuda()

    if args.use_lbfgs:
        optimizer = LBFGSNew(model.parameters(), history_size=7, max_iter=2, line_search_fn=True,batch_mode=True)

    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        if args.arch in ['resnet1202', 'resnet110']:
            # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
            # then switch back. In this setup it will correspond for first epoch.
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1


    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

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

    torch.manual_seed(0)
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    torch.manual_seed(0)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    if args.evaluate:
        validate(val_loader, model, criterion, logfile)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, logfile)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, logfile)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))
    logfile.close()


def train(train_loader, model, criterion, optimizer, epoch, logfile):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

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

        if not args.use_lbfgs:
            # zero gradients
            optimizer.zero_grad()
            # forward+backward optimize
            output = model(input_var)
            loss = criterion(output, target_var)
            loss.backward()
            optimizer.step()
        else:
            if not args.wide_resnet:
                layer1 = torch.cat([x.view(-1) for x in model.layer1.parameters()])
                layer2 = torch.cat([x.view(-1) for x in model.layer2.parameters()])
                layer3 = torch.cat([x.view(-1) for x in model.layer3.parameters()])
                layer4 = torch.cat([x.view(-1) for x in model.layer4.parameters()])

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                output = model(input_var)
                if not args.wide_resnet:
                    l1_penalty = lambda1 * (
                                torch.norm(layer1, 1) + torch.norm(layer2, 1) + torch.norm(layer3, 1) + torch.norm(
                            layer4, 1))
                    l2_penalty = lambda2 * (
                                torch.norm(layer1, 2) + torch.norm(layer2, 2) + torch.norm(layer3, 2) + torch.norm(
                            layer4, 2))
                    loss = criterion(output, target_var) + l1_penalty + l2_penalty
                else:
                    l1_penalty = 0
                    l2_penalty = 0
                    loss = criterion(output, target_var)
                if loss.requires_grad:
                    loss.backward()
                    # print('loss %f l1 %f l2 %f'%(loss,l1_penalty,l2_penalty))
                return loss

            optimizer.step(closure)

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\n'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1), file=logfile)


def validate(val_loader, model, criterion, logfile):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

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
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\n'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1), file=logfile)

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()