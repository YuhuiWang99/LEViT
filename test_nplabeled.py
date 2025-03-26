# -*- coding: utf-8 -*-
from __future__ import print_function, division

from cv2_transform import transforms
from torch.utils.data import DataLoader
import torch

from network.base import BASE
from data_read import ImageTxtDataset

import time, os, sys
import numpy as np
from os import path as osp


def get_data(batch_size, test_set, query_set):
    transform_test = transforms.Compose([
        transforms.Resize(size=(384, 128)),
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
        ff = np.zeros((n, 2048), dtype=np.float32)
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


def get_id(img_path):
    cameras = []
    labels = []
    for path in img_path:
        cameras.append(int(path[0].split('/')[-1].split('_')[1][1]))
        labels.append(path[1])
    return np.array(cameras), np.array(labels)


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
    batch_size = 256
    data_dir = osp.expanduser("/home/wangyh/dataset/reid/cuhk03-np/labeled/")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    test_set = [(osp.join(data_dir,'bounding_box_test',line), int(line.split('_')[0])) for line in os.listdir(data_dir+'bounding_box_test') if ("jpg" in line or "png" in line) and "-1" not in line]
    query_set = [(osp.join(data_dir,'query',line), int(line.split('_')[0])) for line in os.listdir(data_dir+'query') if ("jpg" in line or "png" in line)]

    test_cam, test_label = get_id(test_set)
    query_cam, query_label = get_id(query_set)

    ######################################################################
    # Load Collected data Trained model
    mod_pth = osp.join('params', 'ema.pth')
    net = BASE(num_features=0, num_classes=767, net="resnet50")
    net.load_state_dict(torch.load(mod_pth))
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

