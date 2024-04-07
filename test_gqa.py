import os
import torch_dipu
import torch.distributed as dist
import torch
import torch.distributed._functional_collectives as funcol
import lightllm.patch_lightllm_ops

import torch.nn as nn

def ascend_incre_attention_kernel(q, k, v, int_index_list, max_seq_length):
    return torch.ops.lightllm.flash_attention_inference.default(q, k, v, int_index_list, max_seq_length)

def ascend_prompt_attention_inference(q, k, v, mask):
    return torch.ops.lightllm.prompt_attention_inference.default(q, k, v, mask)


from typing import List
def test_prefill_gqa():
    q = torch.randn(1, 32, 128, dtype=torch.float16).cuda()
    k = torch.randn(17, 8, 128, dtype=torch.float16).cuda()
    v = torch.randn(17, 8, 128, dtype=torch.float16).cuda()
    idx_in = [17]
    compiled_func = torch.compile(ascend_incre_attention_kernel, backend="ascendgraph", dynamic=False)
    result = compiled_func(q, k, v, idx_in, 17)

    k = torch.repeat_interleave(k,4,dim=1)
    v = torch.repeat_interleave(v,4,dim=1)
    qk = torch.matmul(q.permute(1,0,2), k.permute(1,2,0))
    attention_score = qk
    attention_score = qk / (128 ** 0.5)
    attention_weights = nn.functional.softmax(attention_score, dim=-1)
    res2 = torch.matmul(attention_weights, v.permute(1,0,2))

    return ( torch.allclose(result.view(1,32,128),res2.view(1,32,128), rtol=1e-2, atol = 1e-2))

def test_decode_gqa():
    q = torch.randn(64, 32, 128, dtype=torch.float16).cuda()
    k = torch.randn(64, 8, 128, dtype=torch.float16).cuda()
    v = torch.randn(64, 8, 128, dtype=torch.float16).cuda()
    num_head, head_dim = q.shape[1], q.shape[2]
    kvhead = k.shape[1]
    bs = 1
    seqlen = torch.tensor([64], device='cuda:0', dtype=torch.int32)

    atenmask = torch.tril(torch.ones(seqlen.item(), seqlen.item()), diagonal=0).unsqueeze(0).unsqueeze(0).cuda()
    atenmask = atenmask.masked_fill(atenmask == 0., -999999999999.0)
    atenmask = atenmask.masked_fill(atenmask == 1., 0.0)
    atenmask = atenmask.repeat(bs, num_head, 1, 1)

    mask = torch.tril(torch.ones(seqlen.item(), seqlen.item(), dtype=torch.bool), diagonal=0).cuda()
    mask = mask.repeat(1, 1)
    mask = torch.logical_not(mask)


    a10q = q.float()
    a10k = torch.repeat_interleave(k, num_head // kvhead, dim=1).float()
    a10v = torch.repeat_interleave(v, num_head // kvhead, dim=1).float()

    a10qk = torch.matmul(a10q.permute(1,0,2), a10k.permute(1,2,0))
    attention_score = a10qk / (head_dim ** 0.5)
    attention_weights = nn.functional.softmax(attention_score + atenmask.float(), dim=-1)
    aten_out = torch.matmul(attention_weights, a10v.permute(1,0,2))


    compiled_fn = torch.compile(ascend_prompt_attention_inference, backend='ascendgraph', dynamic=False)
    dicp_out = compiled_fn(q, k, v, mask)

    return (torch.allclose(aten_out.transpose(1,2), dicp_out.float(), rtol=1e-2, atol=1e-2))

if __name__ == "__main__":
    if test_prefill_gqa():
        print("prefill gqa passed")
    else:
        print("prefill gqa failed")

    if test_decode_gqa():
        print("decode gqa passed")
    else:
        print("decode gqa failed")
