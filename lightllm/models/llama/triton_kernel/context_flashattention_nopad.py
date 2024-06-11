import torch

# import triton
# import triton.language as tl
import math
from torch.profiler import record_function


import torch.nn.functional as F
def _torch_context_attention(o, xq, xk, xv, mask, act_seq_len, seqlen, num_head, kv_num_head, head_dim):
    bs = len(act_seq_len)
    xq = xq.view(bs, seqlen, num_head, head_dim)
    keys = xk.view(bs, seqlen, kv_num_head, head_dim)
    values = xv.view(bs, seqlen, kv_num_head, head_dim)
    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0).unsqueeze(0).unsqueeze(0).cuda()
    mask = mask.masked_fill(mask == 0., -100000000.0)
    mask = mask.repeat(bs, num_head, 1, 1)
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
    scores = F.softmax(scores.float() + mask, dim=-1).type_as(xq)
    o.copy_(torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(bs*seqlen, num_head*head_dim))


compiled_context_attention = _torch_context_attention #torch.compile(_torch_context_attention, backend='ascendgraph', dynamic=False)


@record_function('eager_context_attention_kernel')
def context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len, head, kv_head, dim):
    batch = b_start_loc.shape[0]
    return _torch_context_attention(q, k, v, batch, b_seq_len, head, kv_head, dim)
    # batch = b_start_loc.shape[0]
    # for i in range(batch):
    #     start = b_start_loc[i]
    #     end = start + b_seq_len[i]
    #     out[start:end] = compiled_context_attention(q[start:end], k[start:end], v[start:end], 1, int(b_seq_len[i]), head, kv_head, dim)
    # return out
context_attention_fwd = context_attention

prompt_flash_attention = context_attention


def test_context_attention():
    max_len_in_batch = 2
    bs = 2
    head = 32
    kv_head = 32
    dim = 128
    # BND S=1
    q = torch.randn(bs* max_len_in_batch, head* dim, dtype=torch.float16).cuda()
    # SND
    k = torch.randn(bs* max_len_in_batch, kv_head* dim, dtype=torch.float16).cuda()
    v = torch.randn(bs* max_len_in_batch, kv_head* dim, dtype=torch.float16).cuda()

    if q.shape[0] == 1:
        req_to_token_indexs = torch.arange(1026, dtype=torch.int32).reshape(1, 1026).cuda()
        b_req_idx = torch.tensor([0], dtype=torch.int32).cuda()
        b_start_loc = torch.tensor([0], dtype=torch.int32).cuda()
        b_seq_len = torch.tensor([max_len_in_batch], dtype=torch.int32).cuda()
    else:
        b_seq_len = torch.tensor([2, 2], dtype=torch.int32).cuda()
    
    mask = torch.tril(torch.ones(max_len_in_batch, max_len_in_batch, dtype=torch.bool), diagonal=0).cuda()
    mask = mask.repeat(bs, 1, 1)
    mask = torch.logical_not(mask)
    print(mask)
    
    o_ext = torch.empty_like(q)
    ext.prompt_flash_attention(o_ext, q, k, v, mask, b_seq_len.tolist(), max_len_in_batch, head, kv_head, dim)

    print(o_ext.shape)
    o_torch = torch.empty_like(q)
    _torch_context_attention(o_torch, q, k, v, mask, b_seq_len.tolist(), max_len_in_batch, head, kv_head, dim)
    print(o_torch.shape)
    assert torch.allclose(o_torch.cpu(), o_ext.cpu(), rtol=1e-2, atol=1e-2)

    
    # print(torch.allclose(o_torch.cpu(), o_ext.cpu(), rtol=1e-2, atol=1e-2))

if __name__ == "__main__":
    import torch_dipu
    import deeplink_ext.cpp_extensions as ext

    test_context_attention()
