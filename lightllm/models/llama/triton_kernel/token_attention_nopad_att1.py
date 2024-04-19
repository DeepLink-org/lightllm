import math
import torch
from torch.profiler import record_function

arange_tensor = torch.arange(0, 2048).cuda()

def step0(Req_to_tokens, B_req_idx):
    b_loc = Req_to_tokens[B_req_idx]
    return b_loc
opt_step0 = torch.compile(step0, backend='ascendgraph', dynamic=False)

def step1(b_seq_len, max_input_len, current_arange):
    k_loc_index = max_input_len - b_seq_len + current_arange
    return k_loc_index
opt_step1 = torch.compile(step1, backend='ascendgraph', dynamic=False)

def step2(xq, key, dim):
    return torch.matmul(xq, key.transpose(2, 3)) / math.sqrt(dim)
opt_step2 = torch.compile(step2, backend='ascendgraph', dynamic=False)

def step3(input, b_start_loc, current_arange, out):
    out_loc = b_start_loc + current_arange
    out[:, out_loc] = input
    return out
opt_step3 = torch.compile(step3, backend='ascendgraph', dynamic=False)

import pickle


@record_function('_token_attention_entrypoint')
def _token_attention(q, k, out, Req_to_tokens, B_req_idx, b_start_loc, b_seq_len, max_input_len):
    b_loc = opt_step0(Req_to_tokens, B_req_idx)
    b_loc = Req_to_tokens[B_req_idx]
    batch, head, dim = b_loc.shape[0], q.shape[1], q.shape[2]
    xq = q.view(batch, 1, head, dim).transpose(1, 2)
    for i in range(batch):
        current_arange = arange_tensor[:b_seq_len[i]]
        with record_function('opt_step1'):
            k_loc_index = opt_step1(b_seq_len[i], max_input_len, current_arange)
        k_loc = b_loc[i][k_loc_index]
        key = k[k_loc, :].view(1, b_seq_len[i], head, dim).transpose(1, 2)
        with record_function('opt_step2'):
            res = opt_step2(xq[i, :], key, dim)
        res = res.reshape(head, b_seq_len[i]).clone()
        with record_function('opt_step3'):
            res = opt_step3(res, b_start_loc[i], current_arange, out)
    return res

token_att_fwd = _token_attention
