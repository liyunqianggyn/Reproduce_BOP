import argparse
import os
import time
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
from modules import ir_1w1a
from args import args
from binary_neural_network import Bopoptimizer


nonlineaty  =  'Prelu'
signmask_Alltype = ['signonly']  

seed_all = [100]
prun_rate_all = [0]

len_signmaskall = len(signmask_Alltype)
len_prune_rate = len(prun_rate_all)
len_seed = len(seed_all)
Top1_all = {}

best_prec1 = 0


def main():
    global args, best_prec1
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)    
        
    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    

    model.cuda()
    best_prec1 = 0
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
        datasets.CIFAR10(root='/tudelft.net/staff-bulk/ewi/insy/VisionLab/yunqiangli/data/bnn/cifar10/', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/tudelft.net/staff-bulk/ewi/insy/VisionLab/yunqiangli/data/bnn/cifar10/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.masksigntype == 'scalesignmaskhalffilter' or args.masksigntype == 'scalesignmaskhalflayer':
        args.scale_fan = True

    if args.half:
        model.half()
        criterion.half()

    
    parameters = list(model.named_parameters())
    bn_params = [v for n, v in parameters if (("bn" in n) and v.requires_grad) or (("layer" not in n) and v.requires_grad)  or (("prelu" in n) and v.requires_grad) ]
    rest_params = [v for n, v in parameters if ("layer" in n) and ("conv" in n) and v.requires_grad]

    optimizerbn = torch.optim.SGD(bn_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    optimizerours = Bopoptimizer(
        rest_params,  
        ar=args.gamma,
        threshold=args.tr,
    )           

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerbn, args.epochs, eta_min = 0, last_epoch=-1)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    T_min, T_max = 1e-1, 1e1

    def Log_UP(K_min, K_max, epoch):
        Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
        return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / args.epochs * epoch)]).float().cuda()

    print (model.module)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizerbn, epoch)

        t = Log_UP(T_min, T_max, epoch)
        if (t < 1):
            k = 1 / t
        else:
            k = torch.tensor([1]).float().cuda()

        for i in range(3):
            model.module.layer1[i].conv1.k = k
            model.module.layer1[i].conv2.k = k
            model.module.layer1[i].conv1.t = t
            model.module.layer1[i].conv2.t = t
                
            model.module.layer2[i].conv1.k = k
            model.module.layer2[i].conv2.k = k
            model.module.layer2[i].conv1.t = t
            model.module.layer2[i].conv2.t = t
                
            model.module.layer3[i].conv1.k = k
            model.module.layer3[i].conv2.k = k
            model.module.layer3[i].conv1.t = t
            model.module.layer3[i].conv2.t = t

        # train for one epoch
        print('current lr {:.5e}'.format(optimizerbn.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizerbn, optimizerours,  epoch)
#        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        print(' Epoch {}, best Prec@1 {}'
              .format(epoch, best_prec1))
        
        acc1_seed[seed_index, 0] = best_prec1
        
def train(train_loader, model, criterion, optimizerbn, optimizerours, epoch):
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
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)


        # compute gradient and do SGD step
        optimizerbn.zero_grad()
        optimizerours.zero_grad()
        loss.backward()
        optimizerbn.step()
        
        ## Optimizing using Bop
        flips = optimizerours.step()    

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

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

    return top1.avg



def adjust_learning_rate(optimizer, epoch):
    update_list = [150, 250, 320, 370]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

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
#    main()
    for signmask_index in range(len_signmaskall):
        args.masksigntype = signmask_Alltype[signmask_index] 
        print("current training model_type: {} for training".format(args.masksigntype))  
        print("activation: {} for training".format(nonlineaty))      

        for prune_index in range(len_prune_rate):
            args.prune_rate = prun_rate_all[prune_index]
            print("prunerate: {} for training".format(args.prune_rate))
   
            acc1_seed = np.zeros([len_seed, 1])
            
            for seed_index in range(len_seed):
                args.seed = seed_all[seed_index]    
                main() 
          
            Top1_all[signmask_index, prune_index] = acc1_seed
            
