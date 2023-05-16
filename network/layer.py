from __future__ import absolute_import

import torch, random, math, cv2
from torch import nn


class BatchErasing(nn.Module):
    def __init__(self, probability=0.5, smin=0.02, smax=0.4, ratio=0.3, wmax=1.0, type="image"):
        super(BatchErasing, self).__init__()
        self.probability = probability
        self.smin = smin
        self.smax = smax
        self.ratio = ratio
        self.wmax = wmax
        self.type = type

    def forward(self, feature):
        if self.type == "batch":
            if random.uniform(0, 1) <= self.probability:
                height, width = feature.shape[2:]
                area = width * height
                target_area = random.uniform(self.smin, self.smax) * area
                aspect_ratio = random.uniform(self.ratio, 1/self.ratio)
                
                w = min(int(round(math.sqrt(target_area * aspect_ratio))), int(round(width * self.wmax)))
                h = min(int(round(target_area / w)), height)
                x = random.randint(0, height - h)
                y = random.randint(0, width - w)
                feature[:, :, x:x+h, y:y+w] = 0

        elif self.type == "class":
            for i in range(feature.shape[0]//4):
                if random.uniform(0, 1) <= self.probability:
                    height, width = feature.shape[2:]
                    area = width * height
                    target_area = random.uniform(self.smin, self.smax) * area
                    aspect_ratio = random.uniform(self.ratio, 1/self.ratio)
                    
                    w = min(int(round(math.sqrt(target_area * aspect_ratio))), int(round(width * self.wmax)))
                    h = min(int(round(target_area / w)), height)
                    x = random.randint(0, height - h)
                    y = random.randint(0, width - w)
                    feature[i*4:i*4+4, :, x:x+h, y:y+w] = 0

        elif self.type == "level":
            w_list = []
            h_list = []
            x_list = []
            y_list = []
            for i in range(4):
                if random.uniform(0, 1) <= self.probability:
                    height, width = feature.shape[2:]
                    area = width * height
                    target_area = random.uniform(self.smin, self.smax) * area
                    aspect_ratio = random.uniform(self.ratio, 1/self.ratio)
                    
                    w_list.append(min(int(round(math.sqrt(target_area * aspect_ratio))), int(round(width * self.wmax))))
                    h_list.append(min(int(round(target_area / w_list[i])), height))
                    x_list.append(random.randint(0, height - h_list[i]))
                    y_list.append(random.randint(0, width - w_list[i]))
                else:
                    w_list.append(0)
                    h_list.append(0)
                    x_list.append(0)
                    y_list.append(0)

            for i in range(feature.shape[0]//4):
                feature[i*4+0, :, x_list[0]:x_list[0]+h_list[0], y_list[0]:y_list[0]+w_list[0]] = 0
                feature[i*4+1, :, x_list[1]:x_list[1]+h_list[1], y_list[1]:y_list[1]+w_list[1]] = 0
                feature[i*4+2, :, x_list[2]:x_list[2]+h_list[2], y_list[2]:y_list[2]+w_list[2]] = 0
                feature[i*4+3, :, x_list[3]:x_list[3]+h_list[3], y_list[3]:y_list[3]+w_list[3]] = 0
                
        elif self.type == "image":
            for i in range(feature.shape[0]):
                if random.uniform(0, 1) <= self.probability:
                    height, width = feature.shape[2:]
                    area = width * height
                    target_area = random.uniform(self.smin, self.smax) * area
                    aspect_ratio = random.uniform(self.ratio, 1/self.ratio)
                    
                    w = min(int(round(math.sqrt(target_area * aspect_ratio))), int(round(width * self.wmax)))
                    h = min(int(round(target_area / w)), height)
                    x = random.randint(0, height - h)
                    y = random.randint(0, width - w)
                    feature[i, :, x:x+h, y:y+w] = 0
                    
        return feature



class NormLinear(nn.Module):
    def __init__(self, num_features, num_classes, scale):
        super(NormLinear, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(num_features, num_classes), requires_grad = True)

    def forward(self, x):
        x_norm = torch.norm(x, 2, 1, True).clamp(min=1e-12).expand_as(x)
        x_norm = x / x_norm
        w_norm = torch.norm(self.weight, 2, 0, True).clamp(min=1e-12).expand_as(self.weight)
        w_norm = self.weight / w_norm
        output = self.scale * torch.mm(x_norm, w_norm)
        return output
