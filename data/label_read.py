"""Market 1501 Person Re-Identification Dataset."""
import random
from os import path as osp

__all__ = ['LabelList']


def LabelList(root='~/.mxnet/datasets', name='market1501'):
    """Load the Label List for Market 1501 dataset.

    Parameters
    ----------
    ratio : float, default 1
        Split label into train and test.
    root : str, default '~/.mxnet/datasets'
        Path to the folder stored the dataset.
    name : str, default 'market1501'
        Which dataset is used. Only support market 1501 now.
    """
    root = osp.expanduser(root)

    if name == "market1501":
        path = osp.join(root, "Market-1501-v15.09.15")
        train_txt = osp.join(path, "train.txt")
        image_path = osp.join(path, "bounding_box_train")

        item_list = [(osp.join(image_path, line.split()[0]), int(line.split()[1]))
                     for line in open(train_txt).readlines()]

        return item_list, osp.join(path, name+".lmdb")

    elif name == "dukemtmc":
        path = osp.join(root, "DukeMTMC-reID")
        train_txt = osp.join(path, "train.txt")
        image_path = osp.join(path, "bounding_box_train")

        item_list = [(osp.join(image_path, line.split()[0]), int(line.split()[1]))
                     for line in open(train_txt).readlines()]

        return item_list, osp.join(path, name+".lmdb")

    elif name == "npdetected":
        path = osp.join(root, "cuhk03-np", "detected")
        train_txt = osp.join(path, "train.txt")
        image_path = osp.join(path, "bounding_box_train")

        item_list = [(osp.join(image_path, line.split()[0]), int(line.split()[1]))
                     for line in open(train_txt).readlines()]

        return item_list, osp.join(path, name+".lmdb")
        
    elif name == "nplabeled":
        path = osp.join(root, "cuhk03-np", "labeled")
        train_txt = osp.join(path, "train.txt")
        image_path = osp.join(path, "bounding_box_train")

        item_list = [(osp.join(image_path, line.split()[0]), int(line.split()[1]))
                     for line in open(train_txt).readlines()]

        return item_list, osp.join(path, name+".lmdb")

    elif name == "msmt17":
        path = osp.join(root, "MSMT17")
        train_txt = osp.join(path, "train.txt")
        image_path = osp.join(path, "bounding_box_train")

        item_list = [(osp.join(image_path, line.split()[0]), int(line.split()[1]))
                     for line in open(train_txt).readlines()]

        return item_list, osp.join(path, name+".lmdb")

    elif name == "veri776":
        path = osp.join(root, "VeRi")
        train_txt = osp.join(path, "train.txt")
        image_path = osp.join(path, "image_train")

        item_list = [(osp.join(image_path, line.split()[0]), int(line.split()[1]))
                     for line in open(train_txt).readlines()]

        return item_list, osp.join(path, name+".lmdb")
    
    elif name == "vehicleid":
        path = osp.join(root, "VehicleID_V1.0")
        train_txt = osp.join(path, "train.txt")
        image_path = osp.join(path, "image")

        item_list = [(osp.join(image_path, line.split()[0]), int(line.split()[1]))
                     for line in open(train_txt).readlines()]

        return item_list, osp.join(path, name+".lmdb")

    return None, None
