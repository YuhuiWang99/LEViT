from __future__ import division

import argparse, datetime, os, math, copy
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

# import torchvision.transforms as transforms
from cv2_transform import transforms 
from torch.utils.data import DataLoader
import torch
from torch import nn

from network.tripletloss import TripletLoss
from data.data_read import ImageTxtDataset, ImageFolderLMDB
from data.label_read import LabelList
from data.sampler import RandomIdentitySampler
import utils

from network.dbn import DBN


# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--amp', type=bool, default=False)
parser.add_argument('--img-height', type=int, default=384,
                    help='the height of image for input')
parser.add_argument('--img-width', type=int, default=128,
                    help='the width of image for input')
parser.add_argument('--batch-size', type=int, default=128,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-workers', type=int, default=8,
                    help='the number of workers for data loader')
parser.add_argument('--dataset-root', type=str, default="/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/benchmarks/reid",
                    help='the number of workers for data loader')
parser.add_argument('--dataset', type=str, default="market1501",
                    help='the number of workers for data loader, market1501, dukemtmc, npdetected, nplabeled, msmt17, veri776, vehicleid')
parser.add_argument('--model', type=str, default="sbn", help="")
parser.add_argument('--net', type=str, default="small", help="small")
parser.add_argument('--std', type=float, default=0.2, help="std to init the weight and bias in batch norm")
parser.add_argument('--freeze', type=str, default="", help="stem,layer1,layer2,layer3")
parser.add_argument('--gpus', type=str, default="0,1",
                    help='number of gpus to use.')
parser.add_argument('--warmup', type=bool, default=True,
                    help='number of training epochs.')
parser.add_argument('--epochs', type=str, default="5,75")
parser.add_argument('--lr', type=float, default=2.0e-3,
                    help='learning rate. default is 2.0e-3.')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay rate. default is 5e-4.')
parser.add_argument('--seed', type=int, default=613,
                    help='random seed to use. Default=613.')
parser.add_argument('--lr-decay', type=int, default=0.1)
parser.add_argument('--swa', type=bool, default=False, help='swa usage flag (default: off)')
parser.add_argument('--swa-ratio', type=float, default=1.0)
parser.add_argument('--swa-cycle', type=int, default=1)
parser.add_argument('--swa-extra', type=int, default=0)

def get_data_iters(batch_size):
    train_set, train_txt = LabelList(root=opt.dataset_root, name=opt.dataset)

    transform_train = transforms.Compose([
        transforms.Resize((opt.img_height, opt.img_width)),
        transforms.Pad(padding=8),
        transforms.RandomCrop((opt.img_height, opt.img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    # train_imgs = ImageTxtDataset(train_set, transform=transform_train)
    train_imgs = ImageFolderLMDB(train_txt, train_set, transform=transform_train)
    train_data = DataLoader(train_imgs, batch_size, sampler=RandomIdentitySampler(train_set, 4), drop_last=False, num_workers=opt.num_workers)

    return train_data


def adjust_lr(epoch, epochs, opt):
    stop_epoch = int(epochs[1] * opt.swa_ratio)
    if opt.warmup:
        minlr = opt.lr * 0.01
        dlr = (opt.lr - minlr) / (epochs[0] - 1)
        if epoch < epochs[0]:
            lr = minlr + dlr * epoch

    if epoch >= epochs[0] and epoch <= stop_epoch:
        lr = 0.5 * opt.lr * (math.cos(math.pi * (epoch - epochs[0]) / (epochs[1] - epochs[0])) + 1)
    
    if epoch > stop_epoch:
        current_epoch = stop_epoch + (epoch - stop_epoch) % opt.swa_cycle
        lr = 0.5 * opt.lr * (math.cos(math.pi * (current_epoch - epochs[0]) / (epochs[1] - epochs[0])) + 1)
    return lr


def main(net, batch_size, epochs, opt):

    if opt.swa:
        swa_net = copy.deepcopy(net)
        swa_net = nn.DataParallel(swa_net)
        swa_net.cuda()
        net = nn.DataParallel(net)
        net.cuda()
        net.train()
        swa_n = 0
        print('SWA training')
    else:
        net = nn.DataParallel(net)
        net.cuda()
        net.train()
        print('SGD training')
        
    train_data = get_data_iters(batch_size)
    trainer = torch.optim.SGD(params=net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.wd, nesterov=True)
    # trainer = torch.optim.Adam(params=net.parameters(), lr=opt.lr, weight_decay=opt.wd)
    # trainer = torch.optim.AdamW(params=net.parameters(), lr=opt.lr, weight_decay=opt.wd)

    if len(opt.freeze) > 0:
        for name, param in net.named_parameters():
            for l in opt.freeze.split(","):
                if l in name:
                    param.requires_grad = False

    if opt.amp:
        scaler = torch.cuda.amp.GradScaler()

    prev_time = datetime.datetime.now()
    criterion1 = nn.CrossEntropyLoss().cuda()
    criterion2 = TripletLoss().cuda()
    for epoch in range(0, epochs[1] + opt.swa_extra):
        _loss = 0.
        length = len(train_data)
        lr = adjust_lr(epoch, epochs, opt)
        for param_group in trainer.param_groups:
            param_group['lr'] = lr
        trainer.defaults['lr'] = lr

        for data, label in train_data:
            data_list = data.cuda()
            label_list = label.cuda()
            trainer.zero_grad()
            losses = []

            if opt.amp:
                with torch.cuda.amp.autocast():
                    outputs, features = net(data_list)
                    for output in outputs:
                        losses.append(criterion1(output, label_list))
                    for feature in features:
                        losses.append(criterion2(feature.float(), label_list))
                    loss = sum(losses) / len(losses)
                scaler.scale(loss).backward()
                scaler.step(trainer)
                scaler.update()
            else:
                with torch.set_grad_enabled(True):
                    outputs, features = net(data_list)
                    for output in outputs:
                        losses.append(criterion1(output, label_list))
                    for feature in features:
                        losses.append(criterion2(feature, label_list))
                    loss = sum(losses) / len(losses)
                loss.backward()
                trainer.step()

            loss = loss.detach().sum().cpu().numpy()
            if np.isnan(loss):
                return False
            if not np.isinf(loss):
                _loss += loss
            else:
                length -= batch_size

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        __loss = _loss/length

        epoch_str = ("Epoch %d. Train loss: %f, " % (epoch, __loss))

        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.defaults['lr']))

        if opt.swa and epoch > int(epochs[1] * opt.swa_ratio) and (epoch + 1) % opt.swa_cycle == 0:
            print(str(trainer.defaults['lr']))
            utils.moving_average(swa_net, net, 1.0 / (swa_n + 1))
            swa_n += 1
            if epoch == epochs[1] + opt.swa_extra - 1:
                utils.bn_update(train_data, swa_net)

    if not os.path.exists("params"):
        os.mkdir("params")
    torch.save(net.module.state_dict(), 'params/base.params')
    if opt.swa:
        torch.save(swa_net.module.state_dict(), 'params/swa.params')
    return True

if __name__ == '__main__':
    opt = parser.parse_args()
    logging.info(opt)

    batch_size = opt.batch_size
    epochs = [int(i) for i in opt.epochs.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if opt.dataset == "market1501":
        num_classes = 751
    elif opt.dataset == "dukemtmc":
        num_classes = 702
    elif opt.dataset == "npdetected":
        num_classes = 767
    elif opt.dataset == "nplabeled":
        num_classes = 767
    elif opt.dataset == "msmt17":
        num_classes = 1041
    elif opt.dataset == "veri776":
        num_classes = 576
    elif opt.dataset == "vehicleid":
        num_classes = 13164
    while True:
        if opt.model == "dbn":
            net = DBN(num_classes=num_classes, num_parts=[1,2], std=opt.std, net=opt.net)

        if main(net, batch_size, epochs, opt):
            break
        del net
        torch.cuda.empty_cache()

