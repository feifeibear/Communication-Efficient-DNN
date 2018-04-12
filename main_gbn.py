import pdb
import argparse
import os
import time
import logging
from random import uniform
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from ast import literal_eval
from torch.nn.utils import clip_grad_norm
from math import ceil
from math import sqrt
import numpy as np
from prune_utils.pruning import prune_perc, check_sparsity

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR',
                    default='./TrainingResults', help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2048, type=int,
                    metavar='N', help='mini-batch size (default: 2048)')
parser.add_argument('--lr_bb_fix', dest='lr_bb_fix', action='store_true',
                    help='learning rate fix for big batch lr =  lr0*(batch_size/128)**0.5')
parser.add_argument('--no-lr_bb_fix', dest='lr_bb_fix', action='store_false',
                    help='learning rate fix for big batch lr =  lr0*(batch_size/128)**0.5')
parser.set_defaults(lr_bb_fix=True)
parser.add_argument('--regime_bb_fix', dest='regime_bb_fix', action='store_true',
                    help='regime fix for big batch e = e0*(batch_size/128)')
parser.add_argument('--no-regime_bb_fix', dest='regime_bb_fix', action='store_false',
                    help='regime fix for big batch e = e0*(batch_size/128)')
parser.set_defaults(regime_bb_fix=False)
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')

parser.add_argument('-mb', '--mini-batch-size', default=64, type=int,
                    help='mini-mini-batch size (default: 64)')
parser.add_argument('--ghost_batch_size', type=int, default=0,
                    help='used for ghost batch size')
# parser.add_argument('--use_pruning', type=bool, default=False,
#                    help='whether use pruning')
parser.add_argument('--pruning_perc', default=0.1, type=float,
                    help='the percent of pruning gradient')
# parser.add_argument('--use_residue_acc', type=bool, default=False,
#                     help='whether use pruning')

parser.add_argument('--use_residue_acc', dest='use_residue_acc', action='store_true',
                    help='use residue accumulating')
parser.add_argument('--no_use_residue_acc', dest='use_residue_acc', action='store_false',
                    help='do not use residue accumulating')
parser.set_defaults(use_residue_acc=False)

parser.add_argument('--use_pruning', dest='use_pruning', action='store_true',
                    help='use gradient pruning')
parser.add_argument('--no_use_pruning', dest='use_pruning', action='store_false',
                    help='do not use gradient pruning')
parser.set_defaults(use_pruning=False)

parser.add_argument('--use_warmup', dest='use_warmup', action='store_true',
                    help='use warm up')
parser.add_argument('--no_use_warmup', dest='use_warmup', action='store_false',
                    help='do not use warm up')
parser.set_defaults(use_warmup=False)

parser.add_argument('--use_sync', dest='use_sync', action='store_true',
                    help='synchronize all parameters every sync_interval steps')
parser.add_argument('--no_use_sync', dest='use_sync', action='store_false',
                    help='synchronize all parameters every sync_interval steps')
parser.set_defaults(use_sync=False)

parser.add_argument('--sync_interval', default=100, type=int,
                    help='sync interval (default: 100)')

parser.add_argument('--use_debug', dest='use_debug', action='store_true',
                    help='to debug')
parser.add_argument('--no_use_debug', dest='use_debug', action='store_false',
                    help='no debug')
parser.set_defaults(use_debug=False)



def main():
    torch.manual_seed(123)
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    if args.regime_bb_fix:
            args.epochs *= ceil(args.batch_size / args.mini_batch_size)

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        torch.cuda.manual_seed(123)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
    adapted_regime = {}
    for e, v in regime.items():
        if args.lr_bb_fix and 'lr' in v:
            v['lr'] *= (args.batch_size / args.mini_batch_size) ** 0.5
        if args.regime_bb_fix:
            e *= ceil(args.batch_size / args.mini_batch_size)
        adapted_regime[e] = v
    regime = adapted_regime
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    logging.info('training regime: %s', regime)
    print({i: list(w.size())
           for (i, w) in enumerate(list(model.parameters()))})
    init_weights = [w.data.cpu().clone() for w in list(model.parameters())]

    U = [[]]
    V = [[]]
    print("USE_RESACC ", args.use_residue_acc, " USE_PRUNING ", args.use_pruning)
    if args.use_residue_acc:
        ghost_batch_num = args.batch_size // args.mini_batch_size
        print('USE PRUNING :', args.pruning_perc * 100, "%")
        if torch.cuda.is_available():
            U = [[torch.zeros(w.size()).cuda() for w in list(model.parameters())]
                    for i in range(ghost_batch_num)]
            V = [[torch.zeros(w.size()).cuda() for w in list(model.parameters())]
                    for i in range(ghost_batch_num)]
        else:
            print("gpu is not avaiable for U, V allocation")



    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)

        # train for one epoch
        train_result = train(train_loader, model, criterion, epoch, optimizer, U, V)

        train_loss, train_prec1, train_prec5, U = [
            train_result[r] for r in ['loss', 'prec1', 'prec5', 'V']]

        # evaluate on validation set
        val_result = validate(val_loader, model, criterion, epoch)
        val_loss, val_prec1, val_prec5 = [val_result[r]
                                          for r in ['loss', 'prec1', 'prec5']]

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'regime': regime
        }, is_best, path=save_path)
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        #Enable to measure more layers
        idxs = [0]#,2,4,6,7,8,9,10]#[0, 12, 45, 63]

        step_dist_epoch = {'step_dist_n%s' % k: (w.data.cpu() - init_weights[k]).norm()
                           for (k, w) in enumerate(list(model.parameters())) if k in idxs}


        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                    train_error5=100 - train_prec5, val_error5=100 - val_prec5,
                    **step_dist_epoch)

        results.plot(x='epoch', y=['train_loss', 'val_loss'],
                     title='Loss', ylabel='loss')
        results.plot(x='epoch', y=['train_error1', 'val_error1'],
                     title='Error@1', ylabel='error %')
        results.plot(x='epoch', y=['train_error5', 'val_error5'],
                     title='Error@5', ylabel='error %')

        for k in idxs:
            results.plot(x='epoch', y=['step_dist_n%s' % k],
                         title='step distance per epoch %s' % k,
                         ylabel='val')

        results.save()


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None, U=None, V=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()


    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda(async=True)
        input_var = Variable(inputs.type(args.type), volatile=not training)
        target_var = Variable(target)

        # compute output
        if not training:
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            losses.update(loss.data[0], input_var.size(0))
            top1.update(prec1[0], input_var.size(0))
            top5.update(prec5[0], input_var.size(0))

        else:

            mini_inputs = input_var.chunk(args.batch_size // args.mini_batch_size)
            mini_targets = target_var.chunk(args.batch_size // args.mini_batch_size)

            #TODO for debug shoul be delete
            if(0 == i):
                print('number of ghost batch is ', len(mini_inputs))

            optimizer.zero_grad()

            # fjr simulate distributed senario
            acc_grad = []
            if args.use_residue_acc:
                if torch.cuda.is_available():
                    acc_grad = [torch.zeros(w.size()).cuda() for w in list(model.parameters())]
                else:
                    print("gpu is not avaiable for acc_grad allocation")

            for k, mini_input_var in enumerate(mini_inputs):
                mini_target_var = mini_targets[k]
                output = model(mini_input_var)
                loss = criterion(output, mini_target_var)

                prec1, prec5 = accuracy(output.data, mini_target_var.data, topk=(1, 5))
                losses.update(loss.data[0], mini_input_var.size(0))
                top1.update(prec1[0], mini_input_var.size(0))
                top5.update(prec5[0], mini_input_var.size(0))

                # compute gradient and do SGD step
                # fjr
                if args.use_residue_acc:
                    # clear grad before accumulating
                    optimizer.zero_grad()

                loss.backward()

                if args.use_residue_acc:
                    if args.use_debug:
                        # TODO debug print U[k], V[k], grad[k]
                        print("=======before pruning=========")
                        for u, v, p in zip(U[k], V[k], model.parameters()):
                            g_len = 1;
                            for dim in p.grad.data.size():
                                g_len *= dim
                            g_flatten = p.grad.data.view(g_len)
                            U_flatten = u.view(g_len)
                            V_flatten = v.view(g_len)
                            for debugIdx in range(10):
                                print("node ", k , ",idx ", debugIdx, ",U ", U_flatten[debugIdx],
                                     ",V ", V_flatten[debugIdx], ",grad ", g_flatten[debugIdx])
                            break
                        print("=======end before pruning=========")

                    # clip_grad_norm(model.parameters(), 5. * (len(mini_inputs) ** -0.5))
                    idx = 0
                    for u, v, p in zip(U[k], V[k], model.parameters()):
                        #prune for conv and linear
                            # DEBUG
                            #print("before pruning : grad_norm : ", p.grad.data.norm(), "residue ", r.norm())
                            #pruning change grad and r
                        if args.use_pruning and len(u.size()) != 1:
                            # TODO how to set m
                            u = 0.0 * u + p.grad.data / len(mini_inputs)
                            v = v + u

                            masks = []
                            if args.use_sync and i % args.sync_interval == 0:
                                masks = 1;
                            else:
                                if args.use_warmup:
                                    if (epoch == 0):
                                        masks = prune_perc(v, 0.25)
                                    elif (epoch == 1):
                                        masks = prune_perc(v, 0.125)
                                    elif (epoch == 2):
                                        masks = prune_perc(v, 0.0675)
                                    elif (epoch == 3):
                                        masks = prune_perc(v, 0.04)
                                    else:
                                        masks = prune_perc(v, 0.01)
                                else:
                                    masks = prune_perc(v, 0.01)

                            p.grad.data = v * masks
                            v = v * (1 - masks)
                            u = u * (1 - masks)

                            # DEBUG
                            if args.use_debug:
                                print("after pruning : grad_norm : ", p.grad.data.norm(), "v", v.norm())
                                print("sparsity of this layer is", check_sparsity(p.grad.data))
                        #new_residue.append(r)
                        acc_grad[idx] += p.grad.data
                        U[k][idx] = u #new_residue
                        V[k][idx] = v
                        idx = idx + 1
                        #print("grad", type(p.grad.data), p.size(), p.grad.data.norm())
                        #print("res", type(r), r.size(), r.norm())

            if args.use_residue_acc:
                for g, p in zip(acc_grad, model.parameters()):
                    # if len(r.size()) != 1:
                    #     print("accumulated sparsity is", check_sparsity(p.grad.data))
                    p.grad.data = g #.div_(len(mini_inputs))

                if args.use_debug:
                    print("=======after pruning=========")
                    for k in range(len(mini_inputs)):
                        for u, v, p in zip(U[k], V[k], model.parameters()):
                            g_len = 1;
                            for dim in p.grad.data.size():
                                g_len *= dim
                            g_flatten = p.grad.data.view(g_len)
                            U_flatten = u.view(g_len)
                            V_flatten = v.view(g_len)
                            for debugIdx in range(10):
                                print("node ", k , ",idx ", debugIdx, ",U ", U_flatten[debugIdx],
                                    ",V ", V_flatten[debugIdx], ",grad ", g_flatten[debugIdx])
                            break
                    print("=======end after pruning=========")
            else:
                for p in model.parameters():
                    p.grad.data.div_(len(mini_inputs))
                clip_grad_norm(model.parameters(), 5.)

            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    return {'loss': losses.avg,
            'prec1': top1.avg,
            'prec5': top5.avg,
            'U' : U,
            'V' : V}


def train(data_loader, model, criterion, epoch, optimizer, U, V):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer, U=U, V=V)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None, U=None, V=None)


if __name__ == '__main__':
    main()
