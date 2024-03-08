import math
import pickle

import torch
import torch_dipu
import torch._dynamo as dynamo
import torch.nn.functional as F

from torch import Tensor

torch._dynamo.config.suppress_errors = False

@torch._custom_op.impl.custom_op('lightllm::prompt_attention_inference')
def prompt_attention_inference(q: Tensor, k: Tensor, v: Tensor, num_head: int, seqlen: Tensor) -> Tensor:
    ...

@prompt_attention_inference.impl_abstract()
def lightllm_prompt_attention_inference_abstract(q: Tensor, k: Tensor, v: Tensor, num_head: int, seqlen: Tensor):
    return torch.empty_like(q)

@prompt_attention_inference.impl(['cpu', 'cuda'])
def lightllm_prompt_attention_inference_impl(q, k, v, num_head, seqlen):
    bs = 2
    head_dim = 128
    
    xq = q.view(bs, seqlen, num_head, head_dim)
    xk = k.view(bs, seqlen, num_head, head_dim)
    xv = v.view(bs, seqlen, num_head, head_dim)

    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=1).unsqueeze(0).unsqueeze(0).cuda()
    mask[mask == 0.] = -100000000.0
    mask = mask.repeat(bs, num_head, 1, 1)

    keys = xk
    values = xv
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)

    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, head_dim)

    return output

def ascend_prompt_attention_inference(q, k, v, num_head, seqlen):
    return torch.ops.lightllm.prompt_attention_inference.default(q, k, v, num_head, seqlen)

def load_tensor(name):
    with open(f'/data2/zhoushenglong/tmp/{name}.pkl', 'rb') as f:
        x = pickle.load(f)
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        return x.cuda()
    return x

q = load_tensor("xq")
k = load_tensor("xk")
v = load_tensor("xv")
bs = 2
num_head = 32
head_dim = 128
seqlen = load_tensor("seqlen")

aten_out = ascend_prompt_attention_inference(q, k , v, bs, num_head, head_dim, seqlen)
# print(aten_out)
# print('aten_out.shape:', aten_out.shape)

compiled_fn = torch.compile(ascend_prompt_attention_inference, backend='ascendgraph', dynamic=False)

dicp_out = compiled_fn(q, k , v, num_head, seqlen)
# print(dicp_out)
# print('dicp_out.shape:', dicp_out.shape)

print(torch.allclose(aten_out, dicp_out))

print(aten_out - dicp_out)
