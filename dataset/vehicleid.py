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
    train_lines = open(osp.join(exdir, "train_test_split", "train_list.txt")).readlines()

    train_list = {}
    for line in train_lines:
        pname, pid = line.strip().split()
        if pid not in train_list:
            train_list[pid] = []
        train_list[pid].append({"pname":pname+".jpg", "pid":pid, "pcam":0})

    with open(osp.join(exdir, 'train.txt'), 'w') as f:
        for i, key in enumerate(train_list):
            for item in train_list[key]:
                f.write(item['pname']+" "+str(i)+" "+str(item["pcam"])+"\n")
    print("Make Label List Done")

def gen_lmdb(exdir):
    lmdb_path = osp.join(exdir, "vehicleid.lmdb")
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir, map_size=8589934592, readonly=False, meminit=False, map_async=True)
    txn = db.begin(write=True)

    item_list = open(osp.join(exdir, "train.txt")).readlines()
    for idx, line in enumerate(item_list):
        pname, pid, pcam = line.strip().split()
        pname = osp.join(exdir, "image", pname)
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
    name = "VehicleID_V1.0"
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