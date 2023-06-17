# Self-supervised Spatiotemporal Learning via Video Clip Order Prediction
import torch
import numpy as np
import math
import itertools


def order_class_index(order):
    """Return the index of the order in its full permutation.

    Args:
        order (tensor): e.g. [0,1,2]
    """
    classes = list(itertools.permutations(list(range(len(order)))))
    return classes.index(tuple(order.tolist()))


def train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch):
    torch.set_grad_enabled(True)
    model.train()

    running_loss = 0.0
    correct = 0
    if args.use_clip:
        tag = 'clip'
    elif args.use_front:
        tag = 'front'
    elif args.use_360:
        tag = '360'
    print("Start training...")

    target_list = []
    pred_list = []
    
    for i, data in enumerate(train_dataloader, 1):
        # get inputs
        if args.dataset == '360x':
            tuple_orders = data[tag]["order"]
        else:
            tuple_clips, tuple_orders = data

        # inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward and backward
        outputs = model(data) # return logits here
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # compute loss and acc
        running_loss += loss.item()
        
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()

        target_list.extend(targets.cpu().detach().numpy())
        pred_list.extend(pts.cpu().detach().numpy())
        
        # print statistics and write summary every N batch
        if i % args.pf == 0:
            avg_loss = running_loss / args.pf
            p = np.asarray(pred_list).flatten()
    
            print("pred_list:", pred_list)

            avg_acc = np.sum(target_list == p)/p.shape[0]
            target_list = []
            pred_list = []
    
            # avg_acc = # correct / (args.pf * args.bs)
            print('[TRAIN] epoch-{}, batch-{}, loss: {:.3f}, acc: {:.3f}'.format(epoch, i, avg_loss, avg_acc))
            step = (epoch -1 ) *len(train_dataloader) + i
            writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
            writer.add_scalar('train/Accuracy', avg_acc, step)
            running_loss = 0.0
            correct = 0

    # summary params and grads per eopch
    for name, param in model.named_parameters():
        try:
            writer.add_histogram('params/{}'.format(name), param, epoch)
            writer.add_histogram('grads/{}'.format(name), param.grad, epoch)
        except:
            pass


def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    if args.use_clip:
        tag = 'clip'
    elif args.use_front:
        tag = 'front'
    elif args.use_360:
        tag = '360'

    target_list = []
    pred_list = []
    for i, data in enumerate(val_dataloader):
        # get inputs
        if args.dataset == '360x':
            tuple_orders = data[tag]["order"]

        else:
            tuple_clips, tuple_orders = data

        # inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        
        # forward
        outputs = model(data) # return logits here
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        
        
        target_list.extend(targets.cpu().detach().numpy())
        pred_list.extend(pts.cpu().detach().numpy())
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
        
        
    target_list = np.asarray(target_list).flatten()
    pred_list = np.asarray(pred_list).flatten()
    
    print("pred_list:", pred_list)
    
    avg_loss = total_loss / len(val_dataloader)
    
    avg_acc = np.sum(target_list == pred_list)/pred_list.shape[0] #correct / len(val_dataloader.dataset)
    
    writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
    writer.add_scalar('val/Accuracy', avg_acc, epoch)

    print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))

    return avg_loss


def test(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    if args.use_clip:
        tag = 'clip'
    elif args.use_front:
        tag = 'front'
    elif args.use_360:
        tag = '360'

    for i, data in enumerate(test_dataloader, 1):
        # get inputs
        if args.dataset == '360x':
            tuple_clips = data[tag]["f"]
            tuple_orders = data[tag]["order"]
        else:
            tuple_clips, tuple_orders = data

        inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(test_dataloader)
    avg_acc = correct / len(test_dataloader.dataset)
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss
