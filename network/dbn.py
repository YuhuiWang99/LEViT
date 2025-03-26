from __future__ import absolute_import

import copy
import torch
from torch import nn
from torch.nn import init
from .layer import BatchDrop, BatchErasing

from .LEViT import LEViT


class DBN(nn.Module):
    def __init__(self, num_classes=751, num_parts=[1,2], std=0.1, net="small", drop=0.0, erasing=0.0):
        super(DBN, self).__init__()
        self.num_parts = num_parts
        if self.training:
            self.batch_erasing = nn.Identity()
            if drop > 0:
                self.batch_erasing = BatchDrop(drop=drop)
            elif erasing > 0:
                self.batch_erasing = BatchErasing(smax=erasing)

        if net == "small":
            base = LEViT(num_classes=1000, stem=16, embed_dim=96, mlp_ratio=1., layers=[4,7,4], num_heads=[1,2,4], split_size=[8,8,8], drop_path=0.05, use_vit=[1,1,0])
            old_checkpoint = torch.load("network/stem16_dim96_ratio1_layers474_heads124_ss777_dp005_vit110.pth", "cpu")["state_dict"]
            feat_num = 384
        if net == "large":
            base = LEViT(num_classes=1000, stem=16, embed_dim=192, mlp_ratio=2., layers=[4,7,4], num_heads=[2,4,8], split_size=[8,8,8], drop_path=0.15, use_vit=[1,1,0])
            old_checkpoint = torch.load("network/stem16_dim192_ratio1_layers474_heads248_ss777_dp010_vit110.pth", "cpu")["state_dict"]
            feat_num = 768


        new_checkpoint = dict()
        for key in old_checkpoint.keys():
            if key.startswith("module."):
                new_checkpoint[key[7:]] = old_checkpoint[key]
            else:
                new_checkpoint[key] = old_checkpoint[key]
        
        base.load_state_dict(new_checkpoint, strict=True) # strict=True
        print("load_state_dict: strict == True")
        embed = 384

        self.stem = base.stem
        self.layer1 = base.stage1
        self.layer2 = base.stage2

        self.branch_1 = base.stage3
        self.branch_1.layers[0].mlp.local_enhance[0].stride = (1, 1)
        self.branch_1.layers[0].skip[0].stride = (1, 1)

        self.branch_2 = copy.deepcopy(self.branch_1)

        self.pool_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.class_list = nn.ModuleList()

        for i in range(len(self.num_parts)):
            self.pool_list.append(nn.AdaptiveAvgPool2d(1))
            bn = nn.BatchNorm1d(feat_num)
            init.normal_(bn.weight, mean=1.0, std=std)
            init.normal_(bn.bias, mean=0.0, std=std)
            self.bn_list.append(bn)

            linear = nn.Linear(feat_num, num_classes, bias=False)
            init.normal_(linear.weight, std=0.001)
            self.class_list.append(linear)

        for i in range(sum(self.num_parts)):
            self.pool_list.append(nn.AdaptiveMaxPool2d(1))
            bn = nn.BatchNorm1d(feat_num)
            init.normal_(bn.weight, mean=1.0, std=std)
            init.normal_(bn.bias, mean=0.0, std=std)
            bn.bias.requires_grad = False
            self.bn_list.append(bn)

            linear = nn.Linear(feat_num, num_classes, bias=False)
            init.normal_(linear.weight, std=0.001)
            self.class_list.append(linear)


    def forward(self, x):
        if self.training:
            x = self.batch_erasing(x)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_chunk = [x1, x2, x1] + list(torch.chunk(x2, dim=2, chunks=self.num_parts[-1]))
        pool_list = []
        bn_list = []
        class_list = []

        for i in range(sum(self.num_parts)+len(self.num_parts)):
            pool = self.pool_list[i](x_chunk[i]).flatten(1)
            pool_list.append(pool)
            bn = self.bn_list[i](pool)
            bn_list.append(bn)
            feat_class = self.class_list[i](bn)
            class_list.append(feat_class)
        if self.training:
            return class_list, bn_list[:2]
        return bn_list


