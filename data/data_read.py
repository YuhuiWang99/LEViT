"""Market 1501 Person Re-Identification Dataset."""
from torch.utils.data import Dataset
from PIL import Image
from cv2_transform.functional import imread
import lmdb
import pickle, six
import numpy as np
import cv2


__all__ = ['ImageTxtDataset', 'ImageFolderLMDB']


class ImageFolderLMDB(Dataset):
    def __init__(self, db_path, items, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        self.items = items
        self.length = len(self.items)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(u"{}".format(idx).encode())
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        # img = Image.open(buf).convert('RGB')
        # img = np.asarray(img)
        img = np.frombuffer(buf.read(), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # load label
        target = unpacked[1]
        label = self.items[idx][1]
        assert target == label

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label)

    def __len__(self):
        return self.length


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
        # img = Image.open(fpath).convert("RGB")
        img = imread(fpath)
        label = self.items[idx][1]
        if len(self.items[idx]) == 6:
            # img = img.crop(self.items[idx][2:])
            x0, y0, x1, y1 = self.items[idx][2:]
            img = img[y0:y1, x0:x1]
        if self._transform is not None:
            img = self._transform(img)
        return img, label

    def __len__(self):
        return len(self.items)
