import torch
import torch.nn as nn
import cuda_linear_fused # 直接导入，不抛ModuleNotFoundError就代表编译成功

class CudaLinearFused(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    
    def forward(self, x):
        ori_shape = x.shape
        H = ori_shape[-1]
        # 压成二维
        x_flat = x.reshape(-1, H)
        out_flat = cuda_linear_fused.linear_forward(x_flat, self.weight, self.bias)
        # 还原原始维度，只替换最后一维
        return out_flat.reshape(*ori_shape[:-1], -1)