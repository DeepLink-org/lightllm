import torch
from torch.profiler import record_function

def step0(Req_to_tokens, B_req_idx):
    b_loc = Req_to_tokens[B_req_idx]
    return b_loc
opt_step0 = torch.compile(step0, backend='ascendgraph', dynamic=False)

def step1(b_seq_len, max_input_len, current_arange):
    k_loc_index = max_input_len - b_seq_len + current_arange
    return k_loc_index
opt_step1 = torch.compile(step1, backend='ascendgraph', dynamic=False)

def step2(x):
    return x.softmax(-1)
opt_step2 = torch.compile(step2, backend='ascendgraph', dynamic=False)

def step3(P, V):
    return torch.matmul(P, V)
opt_step3 = torch.compile(step3, backend='ascendgraph', dynamic=False)

@record_function('_token_softmax_reducev_entrypoint')
def _token_softmax_reducev(logics, v, out, req_to_tokens, b_req_idx, b_start_loc, b_seq_len, max_input_len, other_kv_index):
    with record_function('opt_step0'):
        b_loc = opt_step0(req_to_tokens, b_req_idx)
    batch, head, dim = b_loc.shape[0], v.shape[1], v.shape[2]
    for i in range(batch):
        current_arange = torch.arange(0, b_seq_len[i], device=logics.device)
        with record_function('opt_step1'):
            v_loc_index = opt_step1(b_seq_len[i], max_input_len, current_arange)
        v_loc = b_loc[i][v_loc_index]
        P = logics[:, b_start_loc[i]:b_start_loc[i] + b_seq_len[i]]
        with record_function('opt_step2'):
            P = opt_step2(P)
        P = P.reshape(head, 1, 1, b_seq_len[i]).transpose(0, 1)
        V = v[v_loc, :].view(1, b_seq_len[i], head, dim).transpose(1, 2).clone()
        with record_function('opt_step3'):
            res = opt_step3(P, V)
        out[i, :] = res.view(1, head, dim)
    return out

token_softmax_reducev_fwd = _token_softmax_reducev
