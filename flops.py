from thop import profile, clever_format
import torch
from network.LEViT import LocalEnhanceAttention
from network.dbn import DBN
from network.mgn import MGN
from network.dbn_baseline import DBN as DBN_baseline
# def count_LEViT_cell(m: LocalEnhanceAttention, x: torch.Tensor, y: torch.Tensor):
#     B, C, H, W = x[0].shape
#     H_sp = m.H_sp
#     W_sp = m.W_sp
#     # step 2
#     m.total_ops += 2 * B * C * H * W * (H_sp * W_sp)

if __name__=="__main__":
#     custom_ops = { 
#         LocalEnhanceAttention: count_LEViT_cell,
#     }

    model = DBN(num_classes=751, num_parts=[1,2], net="large")
    model.eval()
    # input = torch.randn(1, 3, 256, 128)
    input = torch.randn(1, 3, 384, 128)
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
