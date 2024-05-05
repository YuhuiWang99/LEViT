# -*- coding: utf-8 -*-
from __future__ import print_function, division

from cv2_transform import transforms
from torch.utils.data import DataLoader
import torch

from network.dbn import DBN
from data_read import ImageTxtDataset

import time, os, sys, copy, random
import numpy as np
from os import path as osp
from collections import defaultdict


def get_data(batch_size, test_set, query_set):
    transform_test = transforms.Compose([
        transforms.Resize(size=(448, 448)),
        transforms.CenterCrop(size=(384, 384)),
        transforms.ToTensor(),
    ])

    test_imgs = ImageTxtDataset(test_set, transform=transform_test)
    query_imgs = ImageTxtDataset(query_set, transform=transform_test)

    test_data = DataLoader(test_imgs, batch_size, shuffle=False, num_workers=8)
    query_data = DataLoader(query_imgs, batch_size, shuffle=False, num_workers=8)
    return test_data, query_data

def extract_feature(net, dataloaders):
    count = 0
    features = []
    for img, _ in dataloaders:
        n = img.shape[0]
        count += n
        print(count)
        ff = np.zeros((n, 384*5), dtype=np.float32)
        for i in range(2):
            if(i==1):
                img = torch.flip(img, [3])
            with torch.no_grad():
                f = torch.cat(net(img.cuda()), dim=1).detach().cpu().numpy()
            ff = ff+f
        features.append(ff)
    features = np.concatenate(features)
    features = features / np.sqrt(np.sum(np.square(features), axis=1, keepdims=True))
    return features

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index))
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

if __name__ == '__main__':
    batch_size = 64
    data_dir = osp.expanduser("/home/wangyh/dataset/reid/University-Release/")
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    query_set = []
    query_label = []
    query_cam = []
    for label in os.listdir(osp.join(data_dir, "test", "query_satellite")):
        for name in os.listdir(osp.join(data_dir, "test", "query_satellite", label)):
            query_set.append([osp.join(data_dir, "test", "query_satellite", label, name), label])
            query_label.append(label)
            query_cam.append(name.split(".")[0])

    test_set = []
    test_label = []
    test_cam = []
    for label in os.listdir(osp.join(data_dir, "test", "gallery_drone")):
        for name in os.listdir(osp.join(data_dir, "test", "gallery_drone", label)):
            test_set.append([osp.join(data_dir, "test", "gallery_drone", label, name), label])
            test_label.append(label)
            test_cam.append(name.split(".")[0])

    print(len(query_set))
    print(len(test_set))
    test_label = np.array(test_label)
    test_cam = np.array(test_cam)
    query_label = np.array(query_label)
    query_cam = np.array(query_cam)

    ######################################################################
    # Load Collected data Trained model
    mod_pth = osp.join('params', 'ema.pth')
    net = DBN(num_classes=701, num_parts=[1,2], net="small")
    net.load_state_dict(torch.load(mod_pth), strict=False)
    net.cuda()
    net.eval()
    
    # Extract feature
    test_loader, query_loader = get_data(batch_size, test_set, query_set)
    print('start test')
    test_feature = extract_feature(net, test_loader)
    print('start query')
    query_feature = extract_feature(net, query_loader)

    num = query_label.size
    dist_all = np.dot(query_feature, test_feature.T)

    CMC = np.zeros(test_label.size)
    ap = 0.0
    for i in range(num):
        cam = query_cam[i]
        label = query_label[i]
        index = np.argsort(-dist_all[i])

        query_index = np.argwhere(test_label==label)
        camera_index = np.argwhere(test_cam==cam)

        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC/num #average CMC
    print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/num))
