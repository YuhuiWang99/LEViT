import os
import os.path as osp
from collections import defaultdict
import numpy as np
import scipy.io as sio

from torch.utils.data import Dataset
from cv2_transform.functional import imread


class ImageFolderCub200(Dataset):
    def __init__(self, db_path, transform=None, instance_num=4):
        self.transform = transform
        self.instance_num = instance_num
        self.label_to_items = defaultdict(list)

        names_lines = open(osp.join(db_path, "images.txt")).readlines()
        labels_lines = open(osp.join(db_path, "image_class_labels.txt")).readlines()
        # train_lines = open(osp.join(db_path, "train_test_split.txt")).readlines()

        # lines = [[name.strip().split()[1], label.strip().split()[1], int(train.strip().split()[1])]for (name, label, train) in zip(names_lines, labels_lines, train_lines)]
        lines = [[name.strip().split()[1], int(label.strip().split()[1]), ]for (name, label) in zip(names_lines, labels_lines)]
        id_list = sorted(list(set([lines[idx][1] for idx in range(len(lines))])))

        count = 0
        for idx in range(len(lines)):
            name, id = lines[idx]
            if id <= 100:
                count += 1
                label = id_list.index(id)
                self.label_to_items[label].append([osp.join(db_path, "images", name), label])

        self.indices = list(range(0, len(id_list)//2))
        self.repeat_iter = count // (len(id_list)//2 * self.instance_num)

        self.shuffle_items()

    def shuffle_items(self):
        self.item_list = []
        for _ in range(self.repeat_iter):
            np.random.shuffle(self.indices)
            for label in self.indices:
                items = self.label_to_items[label]
                idxes = list(range(0, len(items)))
                if len(items) >= self.instance_num:
                    idxes = np.random.choice(idxes, size=self.instance_num, replace=False)
                else:
                    idxes = np.random.choice(idxes, size=self.instance_num, replace=True)
                for idx in idxes:
                    self.item_list.append(items[idx])

    def __getitem__(self, idx):
        path, label = self.item_list[idx]
        img = imread(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.item_list)


class ImageFolderCar196(Dataset):
    def __init__(self, db_path, transform=None, instance_num=4):
        self.transform = transform
        self.instance_num = instance_num
        self.label_to_items = defaultdict(list)

        lines = sio.loadmat(osp.join(db_path, "cars_annos.mat"))["annotations"][0]
        id_list = sorted(list(set([lines[idx][5].squeeze().tolist() for idx in range(len(lines))])))

        count = 0
        for idx in range(len(lines)):
            name, bbox_x1, bbox_y1, bbox_x2, bbox_y2, id, test = lines[idx]
            name = name.squeeze().tolist()
            bbox_x1 = bbox_x1.squeeze().tolist()
            bbox_y1 = bbox_y1.squeeze().tolist()
            bbox_x2 = bbox_x2.squeeze().tolist()
            bbox_y2 = bbox_y2.squeeze().tolist()
            id = id.squeeze().tolist()
            test = test.squeeze().tolist()
            if id <= 98:
                count += 1
                label = id_list.index(id)
                self.label_to_items[label].append([osp.join(db_path, name), label, bbox_x1, bbox_y1, bbox_x2, bbox_y2])
        
        self.indices = list(range(0, len(id_list)//2))
        self.repeat_iter = count // (len(id_list)//2 * self.instance_num)

        self.shuffle_items()

    def shuffle_items(self):
        self.item_list = []
        for _ in range(self.repeat_iter):
            np.random.shuffle(self.indices)
            for label in self.indices:
                items = self.label_to_items[label]
                idxes = list(range(0, len(items)))
                if len(items) >= self.instance_num:
                    idxes = np.random.choice(idxes, size=self.instance_num, replace=False)
                else:
                    idxes = np.random.choice(idxes, size=self.instance_num, replace=True)
                for idx in idxes:
                    self.item_list.append(items[idx])

    def __getitem__(self, idx):
        path, label, bbox_x1, bbox_y1, bbox_x2, bbox_y2 = self.item_list[idx]
        img = imread(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.item_list)

class ImageFolderUniversity(Dataset):
    def __init__(self, db_path, transform=None, instance_num=4):
        self.transform_satellite = transform[0]
        self.transform_drone = transform[1]
        self.instance_num = instance_num
        self.label_to_items_satellite = defaultdict(list)
        self.label_to_items_drone = defaultdict(list)

        id_list = sorted(list(set(os.listdir(osp.join(db_path, "train", "drone")))))

        count = 0
        for id in id_list:
            label = id_list.index(id)
            for name_satellite in os.listdir(osp.join(db_path, "train", "satellite", id)):
                self.label_to_items_satellite[label].append([osp.join(db_path, "train", "satellite", id, name_satellite), label, "satellite"])
                count += 1
            for name_drone in os.listdir(osp.join(db_path, "train", "drone", id)):
                self.label_to_items_drone[label].append([osp.join(db_path, "train", "drone", id, name_drone), label, "drone"])
                count += 1

        self.indices = list(range(0, len(id_list)))
        self.repeat_iter = count // (len(id_list) * self.instance_num)

        self.shuffle_items()

    def shuffle_items(self):
        self.item_list = []
        for _ in range(self.repeat_iter):
            np.random.shuffle(self.indices)
            for label in self.indices:
                self.item_list.append(self.label_to_items_satellite[label][0])
                items = self.label_to_items_drone[label]
                idxes = list(range(0, len(items)))
                if len(items) >= (self.instance_num - 1):
                    idxes = np.random.choice(idxes, size=self.instance_num - 1, replace=False)
                else:
                    idxes = np.random.choice(idxes, size=self.instance_num - 1, replace=True)
                for idx in idxes:
                    self.item_list.append(items[idx])

    def __getitem__(self, idx):
        path, label, view = self.item_list[idx]
        img = imread(path)
        if self.transform_satellite is not None and view == "satellite":
            img = self.transform_satellite(img)
        if self.transform_drone is not None and view == "drone":
            img = self.transform_drone(img)
        return img, label

    def __len__(self):
        return len(self.item_list)


class ImageFolderVehicleID(Dataset):
    def __init__(self, db_path, transform=None, instance_num=4):
        self.transform = transform
        self.instance_num = instance_num
        self.label_to_items = defaultdict(list)

        lines = [x.strip() for x in open(osp.join(db_path, "train_test_split", "train_list.txt"), "r").readlines()]
        id_list = sorted(list(set([line.split()[1] for line in lines])))

        for line in lines:
            name, id = line.split()
            label = id_list.index(id)
            self.label_to_items[label].append([osp.join(db_path, "image", name+".jpg"), label])
        self.indices = list(range(0, len(id_list)))
        self.repeat_iter = len(lines) // (len(id_list) * self.instance_num)

        self.shuffle_items()

    def shuffle_items(self):
        self.item_list = []
        for _ in range(self.repeat_iter):
            np.random.shuffle(self.indices)
            for label in self.indices:
                items = self.label_to_items[label]
                idxes = list(range(0, len(items)))
                if len(items) >= self.instance_num:
                    idxes = np.random.choice(idxes, size=self.instance_num, replace=False)
                else:
                    idxes = np.random.choice(idxes, size=self.instance_num, replace=True)
                for idx in idxes:
                    self.item_list.append(items[idx])

    def __getitem__(self, idx):
        path, label = self.item_list[idx]
        img = imread(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.item_list)


class ImageFolder(Dataset):
    def __init__(self, db_path, transform=None, instance_num=4):
        self.transform = transform
        self.instance_num = instance_num
        self.label_to_items = defaultdict(list)

        names = list(sorted(os.listdir(db_path)))
        id_list = sorted(list(set([name.split("_")[0] for name in names])))

        for name in names:
            label = id_list.index(name.split("_")[0])
            self.label_to_items[label].append([osp.join(db_path, name), label])
        self.indices = list(range(0, len(id_list)))
        self.repeat_iter = len(names) // (len(id_list) * self.instance_num)

        self.shuffle_items()

    def shuffle_items(self):
        self.item_list = []
        for _ in range(self.repeat_iter):
            np.random.shuffle(self.indices)
            for label in self.indices:
                items = self.label_to_items[label]
                idxes = list(range(0, len(items)))
                if len(items) >= self.instance_num:
                    idxes = np.random.choice(idxes, size=self.instance_num, replace=False)
                else:
                    idxes = np.random.choice(idxes, size=self.instance_num, replace=True)
                for idx in idxes:
                    self.item_list.append(items[idx])

    def __getitem__(self, idx):
        try:
            path, label = self.item_list[idx]
            # print(path, label)
            img = imread(path)
            if self.transform is not None:
                img = self.transform(img)
        except Exception as e:
            print("image read error:", e)
            print("file path:", path)
        return img, label

    def __len__(self):
        return len(self.item_list)


class ImageTxtDataset(Dataset):
    """Load the Market 1501 dataset.

    Parameters
    ----------
    items : list
        List for image names and labels.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self, items, transform=None):
        self._transform = transform
        self.items = items

    def __getitem__(self, idx):
        fpath = self.items[idx][0]
        try:
            img = imread(fpath)
        except:
            print(fpath)
        label = self.items[idx][1]
        if self._transform is not None:
            img = self._transform(img)
        return img, label

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    data_dir = "/home/wangyh/dataset/reid"
    dataset = ImageFolder(db_path=os.path.join(data_dir, 'Market-1501-v15.09.15/bounding_box_train'), instance_num=4)
    # dataset = ImageFolderVehicleID(db_path=os.path.join(data_dir, 'VehicleID_V1.0'), instance_num=4)
    # dataset = ImageFolderUniversity(db_path=os.path.join(data_dir, 'University-Release'), instance_num=4)
    # dataset = ImageFolderCar196(db_path=os.path.join(data_dir, 'CARS'), instance_num=4)
    # dataset = ImageFolderCub200(db_path=os.path.join(data_dir, 'CUB_200_2011'), instance_num=4)
    print(len(dataset))
