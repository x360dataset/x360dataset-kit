"""Video clip order prediction."""
import os
import math
import itertools
from parser import get_parser
import time
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from tensorboardX import SummaryWriter


from get_datasets import get_dataset
from get_model import get_model
from functions import *


args = get_parser()
print(vars(args)) 

torch.backends.cudnn.benchmark = True
# Force the pytorch to create context on the specific device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

if args.seed:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

########### model ##############
vcopn = get_model(args)
vcopn.train()

def weight_init(m):

    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


vcopn.apply(weight_init) 



if args.use_clip:
    tag = 'clip'
elif args.use_front:
    tag = 'front'
elif args.use_360:
    tag = '360'
        

if args.mode == 'train':  ########### Train #############
    if args.ckpt:  # resume training
        vcopn.load_state_dict(torch.load(args.ckpt))
        log_dir = os.path.dirname(args.ckpt)
    else:
        if args.desp:
            exp_name = '{}_cl{}_it{}_tl{}_{}_{}'.format(args.model, args.cl, args.it, args.tl, args.desp, time.strftime('%m%d%H%M'))
        else:
            exp_name = '{}_cl{}_it{}_tl{}_{}'.format(args.model, args.cl, args.it, args.tl, time.strftime('%m%d%H%M'))
        log_dir = os.path.join(args.log, exp_name)
    writer = SummaryWriter(log_dir)

    ### dataset ###
    train_dataloader, val_dataloader, test_dataloader = get_dataset(args)

    ### loss funciton, optimizer and scheduler ###
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vcopn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)

    prev_best_val_loss = float('inf')
    prev_best_model_path = None
    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
        time_start = time.time()
        train(args, vcopn, criterion, optimizer, device, train_dataloader, writer, epoch)
        print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
        
        
        val_loss = validate(args, vcopn, criterion, device, val_dataloader, writer, epoch)
        # scheduler.step(val_loss)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        
        # save model every 20 epoches
        if epoch % 20 == 0:
            torch.save(vcopn.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(epoch)))
        # save model for the best val
        if val_loss < prev_best_val_loss:
            model_path = os.path.join(log_dir, 'best_model_{}.pt'.format(epoch))
            torch.save(vcopn.state_dict(), model_path)
            prev_best_val_loss = val_loss
            if prev_best_model_path:
                os.remove(prev_best_model_path)
            prev_best_model_path = model_path


elif args.mode == 'test':  ########### Test #############
    vcopn.load_state_dict(torch.load(args.ckpt))
    test_transforms = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop(112),
        transforms.ToTensor()
    ])
    ### dataset ###
    # train_dataloader, test_dataloader = get_dataset(args)

    train_dataloader, val_dataloader, test_dataloader = get_dataset(args)

    print('TEST video number: {}.'.format(len(test_dataloader)))
    
    criterion = nn.CrossEntropyLoss()
    test(args, vcopn, criterion, device, test_dataloader)


