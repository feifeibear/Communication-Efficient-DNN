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
from math import log
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
parser.add_argument('--resnet_depth', type=int, default=18,
                    help='depth of resnet')
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

parser.add_argument('--use_nesterov', dest='use_nesterov', action='store_true',
                    help='to debug')
parser.add_argument('--no_use_nesterov', dest='use_nesterov', action='store_false',
                    help='no debug')
parser.set_defaults(use_nesterov=False)

parser.add_argument('--use_debug', dest='use_debug', action='store_true',
                    help='to debug')
parser.add_argument('--no_use_debug', dest='use_debug', action='store_false',
                    help='no debug')
parser.set_defaults(use_debug=False)

parser.add_argument('--use_delayed_sgd', dest='use_delayed_sgd', action='store_true',
                    help='to use delayed_sgd')
parser.add_argument('--no_use_delayed_sgd', dest='no_use_delayed_sgd', action='store_false',
                    help='no delayed_sgd')
parser.set_defaults(use_delayed_sgd=False)




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

    # fjr : create N model
    logging.info("creating model %s", args.model)
    node_num = args.batch_size // args.mini_batch_size
    model = [ models.__dict__[args.model] for i in range(node_num)]

    model_config = {'input_size': args.input_size, 'dataset': args.dataset, 'depth': args.resnet_depth}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    # fjr to N models
    # model = model(**model_config)
    for i in range(len(model)):
        # TODO can we call model in this way?
        model[i] = model[i](**model_config)
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        #TODO
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
            #TODO
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    # add index 0
    num_parameters = sum([l.nelement() for l in model[0].parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }

    # we fetch some properties from model,so just use model[0] 
    # fjr add index 0
    transform = getattr(model[0], 'input_transform', default_transform)
    regime = getattr(model[0], 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr
                                           #'momentum': args.momentum,
                                           #'weight_decay': args.weight_decay
                                           }})
    adapted_regime = {}
    logging.info('self-defined momentum : %f, weight_decay : %f', args.momentum, args.weight_decay)
    for e, v in regime.items():
        if args.lr_bb_fix and 'lr' in v:
            # v['lr'] *= (args.batch_size / args.mini_batch_size) ** 0.5
            v['lr'] *= (args.batch_size / 128) ** 0.5
        if args.regime_bb_fix:
            e *= ceil(args.batch_size / args.mini_batch_size)
        adapted_regime[e] = v
    regime = adapted_regime

    # define loss function (criterion) and optimizer
    # fjr add index = 0
    criterion = getattr(model[0], 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)

    #fjr
    for i in range(len(model)):
        model[i].type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        #fjr print results of each model
        validate(val_loader, model, criterion, 0)
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # add N optimizer
    optimizer = [torch.optim.SGD(model[i].parameters(), lr=args.lr) for i in range(len(model))]
    logging.info('training regime: %s', regime)
    print({i: list(w.size())
           for (i, w) in enumerate(list(model[0].parameters()))})
    #TODO check diff models' init_weights is identical?
    init_weights = [w.data.cpu().clone() for w in list(model[0].parameters())]

    U = [[]]
    V = [[]]
    PARAMS_MTX = [[]]
    print("USE_RESACC ", args.use_residue_acc, " USE_PRUNING ", args.use_pruning)
    print("model ", args.model, " use_nesterov ", args.use_nesterov)

    ghost_batch_num = args.batch_size // args.mini_batch_size
    if args.use_residue_acc:
        if torch.cuda.is_available():
            U = [[torch.zeros(w.size()).cuda() for w in list(model[i].parameters())]
                    for i in range(ghost_batch_num)]
            V = [[torch.zeros(w.size()).cuda() for w in list(model[i].parameters())]
                    for i in range(ghost_batch_num)]
        else:
            print("gpu is not avaiable for U, V allocation")
    # fjrcomm.
    # TODO no init
    # PARAMS_MTX = [[[model[i].parameters()]
    PARAMS_MTX = [[[torch.zeros(w.size()).cuda() for w in list(model[i].parameters())]
                    for i in range(ghost_batch_num)]
                     for j in range(ghost_batch_num)]

    groups = []
    lgN = int(log(ghost_batch_num, 2))
    # prepare comm. grooup
    for i in range(lgN):
        visited = [False for ii in range(ghost_batch_num)]
        group = []
        for j in range(ghost_batch_num):
            if(visited[j] == False):
                group.append((j, j + 2**i))
                visited[j] = True
                visited[j + 2**i] = True
        groups.append(group)
    print("Comm. Pairs is as follows")
    print(groups)
    timestamp_Mtx = np.zeros((ghost_batch_num, ghost_batch_num)) - 1

    for epoch in range(args.start_epoch, args.epochs):
        for i in range(len(optimizer)):
            optimizer[i] = adjust_optimizer(optimizer[i], epoch, regime)

        # train for one epoch
        train_result = train(train_loader, model, criterion, epoch, optimizer, U, V, PARAMS_MTX, groups, timestamp_Mtx)

        train_loss, train_prec1, train_prec5, U, V, PARAMS_MTX, timestamp_Mtx = [
            train_result[r] for r in ['loss', 'prec1', 'prec5', 'U', 'V', 'PARAMS_MTX', 'timestamp_Mtx']]

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
            #fjr TODO
            'state_dict': model[0].state_dict(),
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

        # TODO
        step_dist_epoch = {'step_dist_n%s' % k: (w.data.cpu() - init_weights[k]).norm()
                           for (k, w) in enumerate(list(model[0].parameters())) if k in idxs}


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


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None, U=None, V=None, PARAMS_MTX=None, groups=None, timestamp_Mtx=None):
    if args.gpus and len(args.gpus) > 1:
        # fjr TODO we can not do data parallel right for seperate models
        for i in range(len(model)):
            model[i] = torch.nn.DataParallel(model[i], args.gpus)

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
            # TODO only node 0's weight is used to evaluate test data
            # should we average the losses or sum lossess?
            output = model[0](input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            losses.update(loss.data[0], input_var.size(0))
            top1.update(prec1[0], input_var.size(0))
            top5.update(prec5[0], input_var.size(0))

        else:
            mini_inputs = input_var.chunk(args.batch_size // args.mini_batch_size)
            mini_targets = target_var.chunk(args.batch_size // args.mini_batch_size)

            #TODO for debug should be delete
            if(0 == i):
                print('number of ghost batch is ', len(mini_inputs))

            # fjr
            for k in range(len(model)):
                optimizer[k].zero_grad()

            # fjr simulate distributed senario, acc_grad should be a N*N matrix,
            # each elem is parameters of the model
            for k, mini_input_var in enumerate(mini_inputs):
                #print('debug in one it ', k)
                mini_target_var = mini_targets[k]
                output = model[k](mini_input_var)
                loss = criterion(output, mini_target_var)
                #print('debug in one it 1 ', k)

                #TODO update k = 0
                if(0 == k):
                    prec1, prec5 = accuracy(output.data, mini_target_var.data, topk=(1, 5))
                    losses.update(loss.data[0], mini_input_var.size(0))
                    top1.update(prec1[0], mini_input_var.size(0))
                    top5.update(prec5[0], mini_input_var.size(0))

                # compute gradient and do SGD step
                loss.backward()
            # end for k, mini_input_var in enumerate(mini_inputs):

            # a independent version
            # fjr comm.
            N = len(model)

            if args.use_delayed_sgd:
                lgN = int(log(N,2))
                for k in range(N):
                    timestamp_Mtx[k][k] += 1
                    for l, p in enumerate(list(model[k].parameters())):
                        PARAMS_MTX[k][k][l] = p.grad.data.clone()

                # print(timestamp_Mtx)

                # for k in range(N):
                #     optimizer[k].zero_grad()

                # debug_PARAMS_MTX_norm = [[0 for i in range(N)] for j in range(N)]
                # for row in range(N):
                #     for col in range(N):
                #         debug_PARAMS_MTX_norm[row][col] = PARAMS_MTX[row][col][0].norm()
                # print(debug_PARAMS_MTX_norm)

                # update values
                step = i % lgN
                for g in groups[step]:
                    idx_left = g[0]
                    idx_right = g[1]
                    for row in range(N):
                        # TODO copy
                        if(timestamp_Mtx[row][idx_left] < timestamp_Mtx[row][idx_right]):
                            # right -> left
                            for l in range(len(PARAMS_MTX[row][idx_left])):
                                PARAMS_MTX[row][idx_left][l] = PARAMS_MTX[row][idx_right][l].clone()
                        else:
                            # right <- left
                            for l in range(len(PARAMS_MTX[row][idx_left])):
                                PARAMS_MTX[row][idx_right][l] = PARAMS_MTX[row][idx_left][l].clone()
                        timestamp_Mtx[row][idx_left] = timestamp_Mtx[row][idx_right] = \
                    max(timestamp_Mtx[row][idx_left], timestamp_Mtx[row][idx_right])
            else:
                # Sync SGD
                for col in range(N):
                    for row in range(N):
                        for l, p in enumerate(list(model[row].parameters())):
                            #TODO clone?
                            PARAMS_MTX[row][col][l] = p.grad.data.clone()

            # average column to model
            for col in range(N):
                for l, p in enumerate(list(model[col].parameters())):
                    p.grad.data.zero_()
                    for row in range(N):
                        p.grad.data += PARAMS_MTX[row][col][l] / N
                clip_grad_norm(model[col].parameters(), 5.)

            # debug_PARAMS_MTX_norm = [[0 for i in range(N)] for j in range(N)]
            # for row in range(N):
            #     for col in range(N):
            #         debug_PARAMS_MTX_norm[row][col] = PARAMS_MTX[row][col][0].norm()
            # print(debug_PARAMS_MTX_norm)

            for k in range(len(optimizer)):
                optimizer[k].step()

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
            'V' : V,
            'PARAMS_MTX' : PARAMS_MTX,
            'timestamp_Mtx' : timestamp_Mtx}


def train(data_loader, model, criterion, epoch, optimizer, U, V, PARAMS_MTX, groups, timestamp_Mtx):
    # switch to train mode
    for i in range(len(model)):
        model[i].train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer, U=U, V=V,
                   PARAMS_MTX=PARAMS_MTX, groups=groups, timestamp_Mtx=timestamp_Mtx)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    for i in range(len(model)):
        model[i].eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None, U=None, V=None, PARAMS_MTX=None, groups=None, timestamp_Mtx=None)


if __name__ == '__main__':
    main()
