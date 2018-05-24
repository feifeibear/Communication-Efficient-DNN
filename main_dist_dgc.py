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
from prune_utils.pruning import select_top_k, select_top_k_appr, check_sparsity
import horovod.torch as hvd

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



def main():
    hvd.init()
    size = hvd.size()
    local_rank = hvd.local_rank()

    torch.manual_seed(123 + local_rank)
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if hvd.local_rank() == 0:
        setup_logging(os.path.join(save_path, 'log.txt'))
        results_file = os.path.join(save_path, 'results.%s')
        results = ResultsLog(results_file % 'csv', results_file % 'html')

    if local_rank == 0:
        logging.info("saving to %s", save_path)
        logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        torch.cuda.manual_seed(123 + local_rank)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        print(args.gpus[0])
        #torch.cuda.set_device(args.gpus[0])
        if(hvd.local_rank() < len(args.gpus)):
            torch.cuda.set_device(args.gpus[hvd.local_rank()])
        else:
            torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset, 'depth': args.resnet_depth}

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

    U = []
    V = []
    print("USE_RESACC ", args.use_residue_acc, " USE_PRUNING ", args.use_pruning)
    print("model ", args.model, " use_nesterov ", args.use_nesterov)

    if torch.cuda.is_available():
        U = [torch.zeros(w.size()).cuda() for w in list(model.parameters())]
        V = [torch.zeros(w.size()).cuda() for w in list(model.parameters())]
    else:
        print("gpu is not avaiable for U, V allocation")



    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)

        # train for one epoch
        train_result = train(train_loader, model, criterion, epoch, optimizer, U, V)

        train_loss, train_prec1, train_prec5, U, V = [
            train_result[r] for r in ['loss', 'prec1', 'prec5', 'U', 'V']]

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


        if(hvd.local_rank() == 0):
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
    # hvd
    # if args.gpus and len(args.gpus) > 1:
    #    model = torch.nn.DataParallel(model, args.gpus)

    batch_time = AverageMeter()
    pruning_time = AverageMeter()
    select_time = AverageMeter()
    comm_time= AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    masks = [torch.zeros(w.size()).cuda() for w in list(model.parameters())]


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

            # mini_inputs = input_var.chunk(args.batch_size // args.mini_batch_size)
            # mini_targets = target_var.chunk(args.batch_size // args.mini_batch_size)

            #TODO for debug shoul be delete
            optimizer.zero_grad()

            # fjr simulate distributed senario
            # acc_grad = []
            # if torch.cuda.is_available():
            #     acc_grad = [torch.zeros(w.size()).cuda() for w in list(model.parameters())]
            # else:
            #     print("gpu is not avaiable for acc_grad allocation")

            # for k, mini_input_var in enumerate(mini_inputs):
            output = model(input_var)
            loss = criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            losses.update(loss.data[0], input_var.size(0))
            top1.update(prec1[0], input_var.size(0))
            top5.update(prec5[0], input_var.size(0))

            loss.backward()

            if args.use_pruning:
                clip_grad_norm(model.parameters(), 5. * (hvd.size() ** -0.5))

            idx = 0
            for u, v, p in zip(U, V, model.parameters()):
                prune_begin = time.time()
                if args.use_pruning:
                    # TODO how to set rho (momentum)
                    g = p.grad.data / hvd.size()
                    g += p.data * args.weight_decay / hvd.size()
                    if args.use_nesterov:
                        u = args.momentum * (u + g)
                        v = v + u + g
                    else:
                        u = args.momentum * u + g
                        v = v + u

                    select_begin = time.time()
                    ratio = 1 - 0.999
                    if args.use_sync and i % args.sync_interval == 0:
                        masks[idx] = 1;
                    else:
                        if args.use_warmup:
                            # print("iter", i, "node ", k, " pruning layer ", idx)
                            if (epoch == 0):
                                ratio = 1 - 0.75
                            elif (epoch == 1):
                                ratio = 1 - 0.9375
                            elif (epoch == 2):
                                ratio = 1 - 0.984375
                            elif (epoch == 3):
                                ratio = 1 - 0.996
                            else:
                                ratio = 1 - 0.999
                        else:
                            ratio = 1 - 0.999
                        #masks[idx], compressed_val, compressed_idx = select_top_k(v, ratio, masks[idx])
                        masks[idx], compressed_val, compressed_idx = select_top_k_appr(v, ratio, masks[idx])
                    select_time.update(time.time() - select_begin)


                    # TODO check compress
                    p_tmp = v * masks[idx]
                    g_ref = hvd.allreduce(p_tmp, average=False)

                    v = v * (1 - masks[idx])
                    u = u * (1 - masks[idx])

                    comm_begin = time.time()
                    g_size = p.grad.data.size()
                    msg_size = len(compressed_val);
                    # print("compressed_val size is, ", msg_size)
                    gathered_val = hvd.allgather(compressed_val)
                    gathered_idx = hvd.allgather(compressed_idx)
                    p.grad.data = p.grad.data.view(-1)
                    p.grad.data.zero_()
                    # print("gathered_val size is, ", len(gathered_val))
                    # print("val", gathered_val)
                    # print("idx", gathered_idx)
                    for node_idx in range(hvd.size()):
                        p.grad.data[gathered_idx[node_idx*msg_size:(node_idx+1)*msg_size]] += gathered_val[node_idx*msg_size:(node_idx+1)*msg_size]
                    p.grad.data = p.grad.data.view(g_size)

                    comm_time.update(time.time() - comm_begin)

                    U[idx] = u #new_residue
                    V[idx] = v
                else:
                    p.grad.data = p.grad.data / hvd.size()
                    hvd.allreduce_(p.grad.data, average=False)
                idx += 1

                pruning_time.update(time.time() - prune_begin)


            # Master
            idx = 0
            if args.use_pruning:
                pass
            else:
                for p in list(model.parameters()):
                    # print("accumulated sparsity is", check_sparsity(g))
                    # TODO 1. use pytorch sgd optimizer to calculate mom and weight_decay, set mom and wd
                    # used with pruning
                    # TODO 2. implement weight_decay and momentum by myself, set mom=0 and wd = 0
                    # used with baseline
                    g = p.grad.data
                    g += p.data * args.weight_decay
                    V[idx] = args.momentum * V[idx] + g
                    p.grad.data = V[k][idx]
                    clip_grad_norm(model.parameters(), 5.)
                    idx = idx+1

            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if hvd.local_rank() == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Prune {pruning_time.val:.9f} ({pruning_time.avg:.3f})\t'
                             'Select {select_time.val:.9f} ({select_time.avg:.3f})\t'
                             'Communication {comm_time.val:.9f} ({comm_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                 epoch, i, len(data_loader),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 batch_time=batch_time,
                                 data_time=data_time,
                                 pruning_time = pruning_time,
                                 select_time = select_time,
                                 comm_time = comm_time,
                             loss=losses, top1=top1, top5=top5))

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
