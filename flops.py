import torch
from thop import profile, clever_format
from network.LEViT import LocalEnhanceAttention
from network.dbn import DBN

def count_LEViT_cell(m: LocalEnhanceAttention, x: torch.Tensor, y: torch.Tensor):
    B, C, H, W = x[0].shape
    H_sp = m.H_sp
    W_sp = m.W_sp
    # step 2
    m.total_ops += 2 * B * C * H * W * (H_sp * W_sp)


if __name__=="__main__":
    custom_ops = { 
        LocalEnhanceAttention: count_LEViT_cell,
    }

    input = torch.randn(1, 3, 384, 128)

    model = DBN(num_classes=751, num_parts=[1,2], std=0.2)
    model.eval()      # Don't forget to call this before inference.

    print(model)
    
    macs, params = profile(model, inputs=(input, ), custom_ops=custom_ops)
    macs, params = clever_format([macs, params], "%.3f")

    print('Flops:  ', macs)
    print('Params: ', params)


