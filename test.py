import math
import pickle

import torch
import torch_dipu
import torch._dynamo as dynamo
import torch.nn.functional as F

from torch import Tensor

torch._dynamo.config.suppress_errors = False

import time

a = torch.randn(1, 4096).cuda()
b = torch.randn(4096, 4096).cuda()
c = torch.randn(4096, 2 * 4096).cuda()

def func1(a, b):
    res1 = torch.mm(a, b)
    res2 = torch.mm(a, b)
    res3 = torch.mm(a, b)
    res4 = torch.mm(a, b)
    res5 = torch.mm(a, b)
    res6 = torch.mm(a, b)
    res7 = torch.mm(a, b)
    res8 = torch.mm(a, b)
    return res1, res2, res3, res4, res5, res6, res7, res8
    # return res1, res2

def func2(a, c):
    # 等价于两次 torch.mm(a, b)
    res1 = torch.mm(a, c) 
    res2 = torch.mm(a, c)
    res3- = torch.mm(a, c)
    res4 = torch.mm(a, c)
    return res1, res2, res3, res4
    # return res1
    
compiled_func1 = torch.compile(func1, backend='ascendgraph', dynamic=False)
compiled_func2 = torch.compile(func2, backend='ascendgraph', dynamic=False)

time1 = time.time()
path = "/data2/zhoushenglong/lightllm/"
with torch_dipu.profiler.NativeProfile(path, with_stack=False):
# with torch.autograd.profiler.profile(with_stack=True, with_modules=True) as prof:
    t = compiled_func1(a, b)
# output_path = "/data2/zhoushenglong/before_cat"
# prof.export_chrome_trace(output_path)
time2 = time.time()
total_time1 = time2 - time1
print("mm1: ",total_time1, flush=True)

time3 = time.time()
path = "/data2/zhoushenglong/lightllm/"
with torch_dipu.profiler.NativeProfile(path, with_stack=False):
# with torch.autograd.profiler.profile(with_stack=True, with_modules=True) as prof:
    t = compiled_func2(a, c)
# output_path = "/data2/zhoushenglong/after_cat"
# prof.export_chrome_trace(output_path)
time4 = time.time()
total_time2 = time4 - time3
print("mm2: ",total_time2, flush=True)

print("ratio: ", total_time2 / total_time1)



