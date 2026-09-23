import torch
import torch.nn as nn
import cuda_linear # 直接导入，不抛ModuleNotFoundError就代表编译成功

N = 16
H = 128
V = 256

x = torch.randn(N, H, device="cuda", dtype=torch.float32)
w = torch.randn(V, H, device="cuda", dtype=torch.float32)
b = torch.randn(V, device="cuda", dtype=torch.float32)

out = cuda_linear.linear_forward(x, w, b)
print("输出shape：", out.shape)