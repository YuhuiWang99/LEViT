from __future__ import absolute_import

import torch, random, math
from torch import nn
from torch.nn.functional import normalize
from torch.nn import init

class BatchDrop(nn.Module):
    def __init__(self, drop=0.0):
        super(BatchDrop, self).__init__()
        self.drop = drop

    def forward(self, feature):
        f_h_list = []
        for f_h in torch.chunk(feature, chunks=1, dim=3):
            f_h_list.append(self._forward(f_h))
        return torch.cat(f_h_list, dim=3)

    def _forward(self, feature):
        height, width = feature.shape[2:]
        for i in range(feature.shape[0]):
            y = random.randint(0, height)
            x = random.randint(0, width)
            drop_area = int(round(height * width * self.drop))
            length = int(round(math.sqrt(drop_area)))

            y1 = max(0, int(round(y - length // 2)))
            y2 = min(height, int(round(y + length // 2)))
            x1 = max(0, int(round(x - length // 2)))
            x2 = min(width, int(round(x + length // 2)))

            feature[i, :, y1:y2, x1:x2] = 0

        return feature


class BatchErasing(nn.Module):
    def __init__(self, probability=0.5, smin=0.02, smax=0.4, ratio=0.3, wmax=1.0):
        super(BatchErasing, self).__init__()
        self.probability = probability
        self.smin = smin
        self.smax = smax
        self.ratio = ratio
        self.wmax = wmax

    def forward(self, feature):
        height, width = feature.shape[2:]
        for i in range(feature.shape[0]):
            if random.uniform(0, 1) <= self.probability:
                area = width * height
                target_area = random.uniform(self.smin, self.smax) * area
                aspect_ratio = random.uniform(self.ratio, 1/self.ratio)
                    
                w = min(int(round(math.sqrt(target_area * aspect_ratio))), int(round(width * self.wmax)))
                h = min(int(round(target_area / w)), height)
                x = random.randint(0, height - h)
                y = random.randint(0, width - w)
                feature[i, :, x:x+h, y:y+w] = 0
                    
        return feature


class BatchCutout(nn.Module):
    def __init__(self, length=16, value=0):
        super(BatchCutout, self).__init__()
        self.length = length
        self.value = value

    def forward(self, feature):
        height, width = feature.shape[2:]
        for i in range(feature.shape[0]):
            y = random.randint(0, height)
            x = random.randint(0, width)
            
            y1 = max(0, int(round(y - self.length // 2)))
            y2 = min(height, int(round(y + self.length // 2)))
            x1 = max(0, int(round(x - self.length // 2)))
            x2 = min(width, int(round(x + self.length // 2)))
            
            feature[i, :, y1:y2, x1:x2] = self.value
                    
        return feature


class BatchHide(nn.Module):
    def __init__(self, probability=0.5, grid_sizes=[0,16,32,44,56]):
        super(BatchHide, self).__init__()
        self.probability = probability
        self.grid_sizes = grid_sizes

    def forward(self, feature):
        height, width = feature.shape[2:]
        for i in range(feature.shape[0]):
            grid_size = self.grid_sizes[random.randint(0, len(self.grid_sizes) - 1)]
            if grid_size > 0:
                for y in range(0, height, grid_size):
                    for x in range(0, width, grid_size):
                        y_end = min(height, y + grid_size)
                        x_end = min(width, x + grid_size)
                        if random.uniform(0, 1) <= self.probability:
                            feature[i, :, y:y_end, x:x_end] = 0

        return feature


class GridMask(nn.Module):
    def __init__(self, probability=0.5, ratio=0.5):
        super(GridMask, self).__init__()
        self.ratio = ratio
        self.probability = probability

    def forward(self, feature):
        height, width = feature.shape[2:]
        for i in range(feature.shape[0]):
            if random.uniform(0, 1) <= self.probability:
                d = random.randint(2, min(height, width))
                hh = int(1.5 * height)
                ww = int(1.5 * width)
                l = int(d * self.ratio)

                mask = torch.ones((1, hh, ww), device=feature.device, dtype= feature.dtype)
                st_h = random.randint(0, d)
                st_w = random.randint(0, d)

                for j in range(hh // d):
                    s = d * j + st_h
                    e = min(s + l, hh)
                    mask[:, s:e, :] = 0

                for j in range(ww // d):
                    s = d * j + st_w
                    e = min(s + l, ww)
                    mask[:, :, s:e] = 0

                mask = 1 - mask
                mask = mask[:, (hh - height) // 2:(hh - height) // 2 + height, (ww - width) // 2:(ww - width) // 2 + width]
                feature[i] = feature[i] * mask
        return feature
