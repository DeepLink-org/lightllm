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
