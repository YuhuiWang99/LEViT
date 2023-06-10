#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import json, os
from os import path as osp
from zipfile import ZipFile
import lmdb
import pickle


def extract(fpath, root):
    print("Extracting zip file")
    with ZipFile(fpath) as z:
        z.extractall(path=root)
    print("Extracting Done")

def make_list(exdir):
    train_dir = osp.join(exdir, "bounding_box_train")
    train_list = {}
    for _, _, files in os.walk(train_dir, topdown=False):
        for pname in files:
            if '.jpg' in pname:
                pname_split = pname.split('_')
                pid = pname_split[0]
                pcam = pname_split[1][1:]
                if pid not in train_list:
                    train_list[pid] = []
                train_list[pid].append({"pname":pname, "pid":pid, "pcam":pcam})


    with open(osp.join(exdir, 'train.txt'), 'w') as f:
        for i, key in enumerate(train_list):
            for item in train_list[key]:
                f.write(item['pname']+" "+str(i)+" "+item["pcam"]+"\n")
    print("Make Label List Done")

def gen_lmdb(exdir):
    lmdb_path = osp.join(exdir, "msmt17.lmdb")
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir, map_size=1073741824, readonly=False, meminit=False, map_async=True)
    txn = db.begin(write=True)

    item_list = open(osp.join(exdir, "train.txt")).readlines()
    for idx, line in enumerate(item_list):
        pname, pid, pcam = line.strip().split()
        pname = osp.join(exdir, "bounding_box_train", pname)
        pid = int(pid)
        pcam = int(pcam)
        with open(pname, 'rb') as f:
            bin_data = f.read()
        txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps((bin_data, pid, pcam)))
        if idx % 1000 == 0:
            print("[%d/%d]" % (idx, len(item_list)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    print("Flushing database ...")
    db.sync()
    db.close()


def main():
    name = "MSMT17"
    root = osp.expanduser("./")
    if not os.path.exists(root):
        os.mkdir(root)
    fpath = osp.join(root, name+'.zip')
    exdir = osp.join(root, name)

    if os.path.exists(fpath):
        if not osp.isdir(exdir):
            extract(fpath, root)
            make_list(exdir)

    gen_lmdb(exdir)
    

if __name__ == '__main__':
    main()