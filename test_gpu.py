import random
import time
import torch

# 获取可用的 GPU 数量
num_gpu = torch.cuda.device_count()

m = 3 * 10 ** 8
seg = 0.0016

# m = 1 * 10 ** 8
# seg = 0.001

# 遍历所有可用的 GPU
while 1:
    if num_gpu > 1:
        for i in range(4, 8):
            device = torch.device(f"cuda:{i}")
            shape = (m, )
            x = torch.randn(shape, device=device)
            y = torch.randn(shape, device=device)
            z = x * y + torch.exp(x) - torch.log(y.abs() + 1e-8)
            time.sleep(seg)
    else:
        device = torch.device(f"cuda:0")
        shape = (m, )
        x = torch.randn(shape, device=device)
        y = torch.randn(shape, device=device)
        z = x * y + torch.exp(x) - torch.log(y.abs() + 1e-8)
        time.sleep(seg)
