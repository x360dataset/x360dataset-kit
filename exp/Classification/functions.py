

import numpy as np
import torch
import torch.nn as nn




def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = nn.CrossEntropyLoss()
    #  nn.BCEWithLogitsLoss()


    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    for step, input_dict in enumerate(dataloader):


        label = input_dict['cls'].to(device, dtype=torch.long)
        # labels=torch.tensor(labels)
        output_dict = model(input_dict, device)

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend


        fused_loss = criterion(output_dict["fused"], label)
        
        if args.use_directional_audio:
            loss_at = criterion(output_dict["at_out"], label)
            
        loss_v = 0
        loss_a = 0

        if args.aux_loss:
            if args.use_front:
                loss_v = criterion(output_dict['v_front'], label)
                if args.use_audio:
                    loss_a = criterion(output_dict['a_front'], label)

            if args.use_360:
                loss_v = loss_v + criterion(output_dict['v_360'], label)
                if args.use_audio:
                    loss_a = loss_a + criterion(output_dict['a_360'], label)


            if args.use_clip:
                loss_v = loss_v + criterion(output_dict['v_clip'], label)
                if args.use_audio:
                    loss_a = loss_a + criterion(output_dict['a_clip'], label)

            # lay down - (loss/ ((loss/target) + self.eps).detach() / self.opt['loss']['lmk'])
            eps = 1e-6
            total_loss = fused_loss + \
                         loss_v / (loss_v / fused_loss + eps).detach()

            if args.use_audio:
                total_loss = total_loss + \
                          loss_a / (loss_a / fused_loss  + eps).detach()
            
            if args.use_directional_audio:
                total_loss = total_loss + \
                          loss_at / (loss_at / fused_loss + eps).detach()

        total_loss.backward()

        optimizer.step()

        _loss += total_loss.item()
        if args.aux_loss:
            if args.use_audio:
                _loss_a += loss_a.item()
            _loss_v += loss_v.item()

    scheduler.step()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)



import os
from glob import glob

root = '/bask/projects/j/jiaoj-3d-vision/360XProject/Data'

folder = glob(os.path.join(root, "Inside", "*")) + \
          glob(os.path.join(root, "Outside", "*"))

class_folder = glob(os.path.join(root, "Inside", "*")) + \
               glob(os.path.join(root, "Outside", "*"))


def process_class_dict(class_folder):
    class2num, num2class = {}, {}
    id = 0
    for each_class in class_folder:
        class_name = each_class.split("/")[-1]
        if class_name not in class2num:
            num2class[id] = class_name
            class2num[class_name] = id
            id += 1

    return class2num, num2class


class2num, num2class = process_class_dict(class_folder)

def process_result_valid(output_dict, label, acc, args, softmax, num):
    
    prediction = {"total": softmax(output_dict["fused"]),
                  "ensemble": softmax(output_dict["fused"])}

    if args.use_front:
        prediction['v_front'] = softmax(output_dict['v_front'])
        prediction['ensemble'] = prediction['ensemble'] + prediction['v_front']

        if args.use_audio:
            prediction['a_front'] = softmax(output_dict['a_front'])
            prediction['ensemble'] = prediction['ensemble'] + prediction['a_front']

    if args.use_360:
        prediction['v_360'] = softmax(output_dict['v_360'])
        prediction['ensemble'] = prediction['ensemble'] + prediction['v_360']

        if args.use_audio:
            prediction['a_360'] = softmax(output_dict['a_360'])
            prediction['ensemble'] = prediction['ensemble'] + prediction['a_360']

    if args.use_clip:
        prediction['v_clip'] = softmax(output_dict['v_clip'])
        prediction['ensemble'] = prediction['ensemble'] + prediction['v_clip']

        if args.use_audio:
            prediction['a_clip'] = softmax(output_dict['a_clip'])
            prediction['ensemble'] = prediction['ensemble'] + prediction['a_clip']

    for i in range(label.shape[0]):

        # v = np.argmax(pred_v[i].cpu().data.numpy())
        # a = np.argmax(pred_a[i].cpu().data.numpy())
        num[label[i]] += 1.0
        label_t = np.asarray(label[i].cpu())

        print("label:", num2class[int(label_t)], ", prediction:",
              [num2class[m] for m in
               prediction['total'][i].cpu().data.numpy().argsort()[-3:][::-1]])

        # np.argpartition(a, -4)[-4:]
        # pdb.set_trace()
        if label_t == np.argmax(prediction['total'][i].cpu().data.numpy()):
            acc["total"][label[i]] += 1.0

        
        if label_t == np.argmax(prediction['ensemble'][i].cpu().data.numpy()):
            acc["ensemble"][label[i]] += 1.0
        
        
            
        if args.use_front:
            if label_t == np.argmax(prediction['v_front'][i].cpu().data.numpy()):
                acc["v_front"][label[i]] += 1.0
            if args.use_audio:
                if label_t == np.argmax(prediction['a_front'][i].cpu().data.numpy()):
                    acc["a_front"][label[i]] += 1.0

        if args.use_360:
            if label_t == np.argmax(prediction['v_360'][i].cpu().data.numpy()):
                acc["v_360"][label[i]] += 1.0
            if args.use_audio:
                if label_t == np.argmax(prediction['a_360'][i].cpu().data.numpy()):
                    acc["a_360"][label[i]] += 1.0

        if args.use_clip:
            if label_t == np.argmax(prediction['v_clip'][i].cpu().data.numpy()):
                acc["v_clip"][label[i]] += 1.0
            if args.use_audio:
                if label_t == np.argmax(prediction['a_clip'][i].cpu().data.numpy()):
                    acc["a_clip"][label[i]] += 1.0

    return acc

def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == '360x':
        n_classes = 21
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    def zero_list():
        return [0.0 for _ in range(n_classes)]

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = zero_list()
        acc = {"total": zero_list(), "ensemble": zero_list(),
               "v_front": zero_list(), "a_front": zero_list(),
                "v_360": zero_list(), "a_360": zero_list(),
                "v_clip": zero_list(), "a_clip": zero_list()}


        for step, input_dict in enumerate(dataloader):

            label = input_dict['cls'].to(device, dtype=torch.long)

            output_dict = model(input_dict, device)

            acc = process_result_valid(output_dict, label, acc, args, softmax, num)

        acc_sum = {}
        for k in acc.keys():
            acc_sum[k] = sum(acc[k]) / sum(num)

    return acc_sum





def valid_folder(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == '360x':
        n_classes = 22
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    def zero_list():
        return np.array([0.0 for _ in range(n_classes)])

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = zero_list()
        acc = {"total": zero_list(),
               "v_front": zero_list(), "a_front": zero_list(),
                "v_360": zero_list(), "a_360": zero_list(),
                "v_clip": zero_list(), "a_clip": zero_list()}


        for step, input_dict in enumerate(dataloader):

            label = input_dict['cls'].to(device, dtype=torch.long)

            prediction = {"total": zero_list(), "v_front": zero_list(),
                          "a_front": zero_list(),
                            "v_360": zero_list(), "a_360": zero_list(),
                            "v_clip": zero_list(), "a_clip": zero_list()}



            for idx in np.arange(input_dict["length"].cpu().numpy()[0]):
                output_dict = model(input_dict[idx], device)

                prediction["total"] += softmax(output_dict["fused"]).cpu().numpy()[0]

                if args.aux_loss:
                    if args.use_front:
                        prediction['v_front'] += softmax(output_dict['v_front']).cpu().numpy()[0]
                        if args.use_audio:
                            prediction['a_front'] += softmax(output_dict['a_front']).cpu().numpy()[0]

                    if args.use_360:
                        prediction['v_360'] += softmax(output_dict['v_360']).cpu().numpy()[0]
                        if args.use_audio:
                            prediction['a_360'] += softmax(output_dict['a_360']).cpu().numpy()[0]

                    if args.use_clip:
                        prediction['v_clip'] += softmax(output_dict['v_clip']).cpu().numpy()[0]
                        if args.use_audio:
                            prediction['a_clip'] += softmax(output_dict['a_clip']).cpu().numpy()[0]


            # print("prediction[total]:", prediction["total"])

            for i in range(label.shape[0]):


                # v = np.argmax(pred_v[i].cpu().data.numpy())
                # a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0
                label_t = np.asarray(label[i].cpu())


                print("label:",  num2class[int(label_t)], ", prediction:",
                      [num2class[m] for m in
                prediction['total'][i].cpu().data.numpy().argsort()[-3:][::-1]])

                #pdb.set_trace()
                if label_t == np.argmax(prediction['total']):
                    acc["total"][label[i]] += 1.0

                if label_t == np.argmax(prediction['ensemble'][i]):
                    acc["ensemble"][label[i]] += 1.0

                if args.aux_loss:
                    if args.use_front:
                        if label_t == np.argmax(prediction['v_front']):
                            acc["v_front"][label[i]] += 1.0
                        if args.use_audio:
                            if label_t == np.argmax(prediction['a_front']):
                                acc["a_front"][label[i]] += 1.0

                    if args.use_360:
                        if label_t == np.argmax(prediction['v_360']):
                            acc["v_360"][label[i]] += 1.0
                        if args.use_audio:
                            if label_t == np.argmax(prediction['a_360']):
                                acc["a_360"][label[i]] += 1.0

                    if args.use_clip:
                        if label_t == np.argmax(prediction['v_clip']):
                            acc["v_clip"][label[i]] += 1.0
                        if args.use_audio:
                            if label_t == np.argmax(prediction['a_clip']):
                                acc["a_clip"][label[i]] += 1.0


        acc_sum = {}
        for k in acc.keys():
            acc_sum[k] = sum(acc[k]) / sum(num)

    return acc_sum

