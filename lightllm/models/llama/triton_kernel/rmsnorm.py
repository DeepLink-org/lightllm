import torch
from torch.profiler import record_function

def torch_rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=+True) + eps) * weight

compiled_torch_rms_norm = torch.compile(torch_rms_norm, backend='ascendgraph', dynamic=True)
rmsnorm_forward = compiled_torch_rms_norm
