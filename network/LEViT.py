# from audioop import bias
import torch
import torch.nn as nn
from torch.nn import init
from einops import rearrange
from timm.models.layers import DropPath
import torch.nn.functional as F

class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, kernel_size=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2, bias=False)
        self.norm = nn.BatchNorm2d(out_planes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.norm(self.conv(x))
        out = self.act(out)
        return out

class LocalEnhanceMLP(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, stride=1):
        super().__init__()

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_dim, h_dim, 1, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(inplace=True),
        )

        self.local_enhance = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=stride, padding=1, groups=h_dim//4, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(inplace=True)
        )
        
        self.proj_out = nn.Sequential(
            nn.Conv2d(h_dim, out_dim, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        self.stride = stride

    def forward(self, x):
        input = self.proj_in(x)
        input = self.local_enhance(input)
        return self.proj_out(input)


class LocalEnhanceAttention(nn.Module):
    def __init__(self, dim, split_size=7, num_heads=8):
        super().__init__()
        self.dim = dim
        self.split_size = split_size
        self.num_heads = num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.H_sp = self.split_size
        self.W_sp = self.split_size

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

    def forward(self, q, k, v):
        """
        x: B L C
        """

        ### Img2Window
        B, C, H, W = q.shape

        q = rearrange(q, 'b (h d) (hh ws1) (ww ws2) -> b (hh ww) h (ws1 ws2) d', h=self.num_heads, hh=H//self.H_sp, ws1=self.H_sp, ws2=self.W_sp)
        k = rearrange(k, 'b (h d) (hh ws1) (ww ws2) -> b (hh ww) h (ws1 ws2) d', h=self.num_heads, hh=H//self.H_sp, ws1=self.H_sp, ws2=self.W_sp)
        v = rearrange(v, 'b (h d) (hh ws1) (ww ws2) -> b (hh ww) h (ws1 ws2) d', h=self.num_heads, hh=H//self.H_sp, ws1=self.H_sp, ws2=self.W_sp)

        dots = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        dots = dots * logit_scale

        attn = dots.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b (hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, hh=H//self.H_sp, ws1=self.H_sp, ws2=self.W_sp)

        return out


class LocalEnhanceBlock(nn.Module):
    def __init__(self, dim, out_dim, num_heads, split_size=7, mlp_ratio=1, stride=1, drop_path=0.0, use_vit=True):
        
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.use_vit = use_vit == 1
        if self.use_vit:
            self.q = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim//4, bias=False),
                nn.BatchNorm2d(dim)
            )
            self.k = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim//4, bias=False),
                nn.BatchNorm2d(dim)
            )
            self.v = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim//4, bias=False),
                nn.BatchNorm2d(dim)
            )
            self.v_spe = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim//4, bias=False),
                nn.BatchNorm2d(dim)
            )


            self.attns = LocalEnhanceAttention(dim, split_size=split_size, num_heads=num_heads)
            self.proj = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(dim)
            )
        
        self.mlp = LocalEnhanceMLP(in_dim=dim, h_dim=int(mlp_ratio*out_dim), out_dim=out_dim, stride=stride)
        if stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim//4, bias=False),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim)
            )
        if drop_path == "relu":
            self.drop_path = nn.ReLU()
        else:
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        if self.use_vit:
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            v_spe = self.v_spe(v)
            x = x + self.drop_path(self.proj(self.attns(q, k, v) + v_spe))

        x = self.skip(x) + self.drop_path(self.mlp(x))
        return x


class StageModule(nn.Module):
    def __init__(self, layers, dim, out_dim, num_heads, split_size, mlp_ratio=1., drop_path=0.0, use_vit=0):
        super().__init__()
        self.layers = []
        for idx in range(layers):
            if idx == 0:
                self.layers.append(LocalEnhanceBlock(dim, out_dim, num_heads, split_size=split_size, mlp_ratio=mlp_ratio, stride=2, drop_path=drop_path[idx], use_vit=use_vit))
            else:
                self.layers.append(LocalEnhanceBlock(out_dim, out_dim, num_heads, split_size=split_size, mlp_ratio=mlp_ratio, stride=1, drop_path=drop_path[idx], use_vit=use_vit))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)



class LEViT(nn.Module):
    def __init__(self, num_classes=1000, stem=12, embed_dim=48, mlp_ratio=1., layers=[4,7,4], num_heads=[1,2,4,8], split_size=[7,7,7,7], drop_path=0.0, use_vit=[0,0,0,0]):
        super().__init__()
        self.num_classes = num_classes
        
        if drop_path == "relu":
            dpr = ["relu" for x in range(0, sum(layers))]
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path, sum(layers))]

        self.stem = nn.Sequential(
            ConvX(3, stem, kernel_size=3, stride=2),
            ConvX(stem, stem, kernel_size=3, stride=1),
            ConvX(stem, stem*2, kernel_size=3, stride=2),
            ConvX(stem*2, stem*2, kernel_size=3, stride=1),
        )

        self.stage1 = StageModule(layers[0], stem*2, embed_dim, num_heads[0], split_size[0], mlp_ratio=mlp_ratio, drop_path=dpr[:layers[0]], use_vit=use_vit[0])
        self.stage2 = StageModule(layers[1], embed_dim, embed_dim*2, num_heads[1], split_size[1], mlp_ratio=mlp_ratio, drop_path=dpr[layers[0]:sum(layers[:2])], use_vit=use_vit[1])
        self.stage3 = StageModule(layers[2], embed_dim*2, embed_dim*4, num_heads[2], split_size[2], mlp_ratio=mlp_ratio, drop_path=dpr[sum(layers[:2]):], use_vit=use_vit[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Classifier head
        self.head = nn.Linear(embed_dim*4, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x) # x4

        x = self.stage1(x) # x8
        x = self.stage2(x) # x16
        x = self.stage3(x) # x32

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    from thop import profile, clever_format
    model = LEViT(num_classes=1000, stem=16, embed_dim=96, mlp_ratio=1., layers=[4,7,4], num_heads=[1,2,4], split_size=[8,8,8], drop_path=0.05, use_vit=[1,1,0])

    model.eval()
    input = torch.randn(1, 3, 384, 128)
    out = model(input)
    print(out.shape)

    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
