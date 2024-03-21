import math
import pickle

import torch
import torch_dipu
import torch._dynamo as dynamo
import torch.nn.functional as F

from torch import Tensor

torch._dynamo.config.suppress_errors = False

@torch._custom_op.impl.custom_op('lightllm::prompt_attention_inference')
def prompt_attention_inference(q: Tensor, k: Tensor, v: Tensor, num_head: int, seqlen: Tensor, mask: Tensor) -> Tensor:
    ...

@prompt_attention_inference.impl_abstract()
def lightllm_prompt_attention_inference_abstract(q: Tensor, k: Tensor, v: Tensor, num_head: int, seqlen: Tensor,  mask: Tensor):
    return torch.empty_like(q)

@prompt_attention_inference.impl(['cpu', 'cuda'])
def lightllm_prompt_attention_inference_impl(q, k, v, num_head, seqlen, mask):
    bs = 1
    head_dim = 128
    seqlen = seqlen.item()

    xq = q.view(bs, seqlen, num_head, head_dim)
    xk = k.view(bs, seqlen, num_head, head_dim)
    xv = v.view(bs, seqlen, num_head, head_dim)

    # mask = torch.tril(torch.ones(seqlen[0], seqlen[0]), diagonal=1).unsqueeze(0).unsqueeze(0)
    # mask[mask == 0.] = -100000000.0
    # mask = mask.repeat(bs, num_head, 1, 1)

    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0).unsqueeze(0).unsqueeze(0)
    mask = mask.masked_fill(mask == 0., -100000000.0)
    mask = mask.masked_fill(mask == 1., 0.0)
    mask = mask.repeat(bs, num_head, 1, 1)

    # keys = xk
    # values = xv
    xq = xq.transpose(1, 2)
    keys = xk.transpose(1, 2)
    values = xv.transpose(1, 2)

    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    # scores = F.softmax(scores.float() + mask, dim=-1).type_as(xq)

    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, head_dim)

    return output

def ascend_prompt_attention_inference(q, k, v, num_head, seqlen, mask):
    return torch.ops.lightllm.prompt_attention_inference.default(q, k, v, num_head, seqlen, mask)

def load_tensor(name):
    with open(f'/data2/zhoushenglong/tmp/{name}.pkl', 'rb') as f:
        x = pickle.load(f)
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        return x.cuda()
    return x

q = load_tensor("q")
k = load_tensor("k")
v = load_tensor("v")
bs = 1
num_head = 32
head_dim = 128
seqlen = load_tensor("b_seq_len")
start_loc = load_tensor("b_start_loc")


mask = torch.tril(torch.ones(seqlen.item(), seqlen.item()), diagonal=1).unsqueeze(0).unsqueeze(0).cuda()
mask[mask == 0.] = -10000000000.0
mask = mask.repeat(bs, num_head, 1, 1)

mask = torch.tril(torch.ones(seqlen.item(), seqlen.item()), diagonal=0).unsqueeze(0).unsqueeze(0).cuda()
mask = mask.masked_fill(mask == 0., -100000000.0)
mask = mask.masked_fill(mask == 1., 0.0)
mask = mask.repeat(bs, num_head, 1, 1)

aten_out = ascend_prompt_attention_inference(q.view(-1, 32, 128),
                                            k.view(-1, 32, 128), 
                                            v.view(-1, 32, 128), 
                                            num_head, seqlen, mask)

# print(aten_out)
# print('aten_out.shape:', aten_out.shape)

compiled_fn = torch.compile(ascend_prompt_attention_inference, backend='ascendgraph', dynamic=False)

# import pdb; pdb.set_trace()

dicp_out = compiled_fn(q, k, v, num_head, seqlen, mask)
# print(dicp_out)
# print('dicp_out.shape:', dicp_out.shape)

# print(aten_out - dicp_out)

t = aten_out - dicp_out

print(t.shape)

for i in range(32):
    print(t[0][i], flush=True)

# print(torch.equal(aten_out, dicp_out))
# print(torch.allclose(aten_out, dicp_out, rtol=1e-02, atol=1e-02, equal_nan=True))
