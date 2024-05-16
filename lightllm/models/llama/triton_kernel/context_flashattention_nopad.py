import torch

# import triton
# import triton.language as tl
import math
from torch.profiler import record_function


import torch.nn.functional as F
def _torch_context_attention(xq, xk, xv, bs, seqlen, num_head, head_dim):
    kv_num_head = xk.shape[1]
    xq = xq.view(bs, seqlen, num_head, head_dim)
    xk = xk.view(bs, seqlen, kv_num_head, head_dim)
    xv = xv.view(bs, seqlen, kv_num_head, head_dim)
    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0).unsqueeze(0).unsqueeze(0).cuda()
    mask = mask.masked_fill(mask == 0., -100000000.0)
    mask = mask.repeat(bs, num_head, 1, 1)
    keys = xk
    values = xv
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
    scores = F.softmax(scores.float() + mask, dim=-1).type_as(xq)
    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, head_dim)
    return output


compiled_context_attention = _torch_context_attention #torch.compile(_torch_context_attention, backend='ascendgraph', dynamic=False)


@record_function('eager_context_attention_kernel')
def context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len):
    batch, head, dim = b_start_loc.shape[0], q.shape[1], q.shape[2]
    for i in range(batch):
        start = b_start_loc[i]
        end = start + b_seq_len[i]
        out[start:end, :] = compiled_context_attention(q[start:end], k[start:end], v[start:end], 1, int(b_seq_len[i]), head, dim)
    return out
context_attention_fwd = context_attention

mask_cache = {}
def fused_context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len):
    batch, head, dim = b_start_loc.shape[0], q.shape[2], q.shape[3]
    numKeyValueHeads = k.shape[2]
    assert k.shape[2] == v.shape[2]
    scale = 1 / math.sqrt(dim)

    mask_key_str = str(batch) + ":" + str(max_input_len)
    if mask_key_str not in mask_cache:
        mask = torch.tril(torch.ones(max_input_len, max_input_len, dtype=torch.bool), diagonal=0).cuda()
        mask = mask.repeat(batch, 1, 1)
        mask = torch.logical_not(mask)
        mask_cache[mask_key_str] = mask
        print(f"cache mask in context attention, batch:seqLen={mask_key_str}")
    
    mask = mask_cache[mask_key_str]
    ext.prompt_flash_attention(
        out.view(batch, max_input_len, head*dim), 
        q.view(batch, max_input_len, head*dim), 
        k.view(batch, max_input_len, head*dim), 
        v.view(batch, max_input_len, head*dim), 
        None, mask, b_seq_len, head, scale, 2147473647, 0, "BSH", numKeyValueHeads)
    return out

def test_context_attention():
    max_len_in_batch = 4
    # BND S=1
    q = torch.randn(1, max_len_in_batch, 32, 128, dtype=torch.float16).cuda()
    # SND
    k = torch.randn(1, max_len_in_batch, 32, 128, dtype=torch.float16).cuda()
    v = torch.randn(1, max_len_in_batch, 32, 128, dtype=torch.float16).cuda()

    if q.shape[0] == 1:
        req_to_token_indexs = torch.arange(1026, dtype=torch.int32).reshape(1, 1026).cuda()
        b_req_idx = torch.tensor([0], dtype=torch.int32).cuda()
        b_start_loc = torch.tensor([0], dtype=torch.int32).cuda()
        b_seq_len = torch.tensor([1], dtype=torch.int32).cuda()
    elif q.shape[0] == 2:
        req_to_token_indexs = torch.zeros((2, 512), dtype=torch.int32).cuda()
        req_to_token_indexs[0, 0:127] = torch.arange(0, 127, 1, dtype=torch.int32).cuda()
        req_to_token_indexs[1, 0:127] = torch.arange(128, 255, 1, dtype=torch.int32).cuda()
        b_req_idx = torch.tensor([0, 1], dtype=torch.int32).cuda()
        b_start_loc = torch.tensor([0, 128], dtype=torch.int32).cuda()
        b_seq_len = torch.tensor([127, 127], dtype=torch.int32).cuda()
    
    other_kv_index = 0
    o_torch = torch.empty_like(q)

    fused_context_attention(q, k, v, o_torch, b_start_loc, b_seq_len.tolist(), max_len_in_batch)
    print(f"out.shape:{o_torch.shape}")
    print(o_torch)

    # o_ext = torch.empty_like(q)
    # ext_token_attention(q, k, v, o_ext, req_to_token_indexs[b_req_idx],
    #                       b_start_loc, b_seq_len, max_len_in_batch, other_kv_index)
    # print(o_ext)
    # assert torch.allclose(o_torch.cpu(), o_ext.cpu(), rtol=1e-2, atol=1e-2)

    
    # print(torch.allclose(o_torch.cpu(), o_ext.cpu(), rtol=1e-2, atol=1e-2))

if __name__ == "__main__":
    import torch_dipu
    import deeplink_ext.cpp_extensions as ext
    # torch.manual_seed(1000)
    # torch.cuda.manual_seed(1000)
    shape1 = 11008
    a1 = torch.randn(20, shape1, dtype=torch.float16, device="cuda")
    a2 = a1[10:13].clone()
    w = torch.randn(shape1, 4096, dtype=torch.float16, device="cuda")
    o1 = torch.mm(a1, w)
    o2 = torch.mm(a2, w)

    print(torch.allclose(o1[10:13].clone(), o2, rtol=1e-6, atol=1e-6))

   # test_context_attention()
