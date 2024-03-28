import math
import torch
from torch.profiler import record_function

arange_tensor = torch.arange(0, 512).cuda()

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

def dump_tensor(x, name):
    with open(f'/tzy/ext_ops/token_attention_inference/{name}.pkl', 'wb') as f:
        if isinstance(x, torch.Tensor):
            pickle.dump(x.cpu(), f)
        else:
            pickle.dump(x, f)

@record_function('_token_attention_entrypoint')
def _token_attention(q, k, out, Req_to_tokens, B_req_idx, b_start_loc, b_seq_len, max_input_len):
    # dump_tensor(q, "q") 
    # dump_tensor(k, "k")
    # dump_tensor(out, "out")
    # dump_tensor(Req_to_tokens, "Req_to_tokens")
    # dump_tensor(B_req_idx, "B_req_idx")
    # dump_tensor(b_start_loc, "b_start_loc")
    # dump_tensor(b_seq_len, "b_seq_len")
    # dump_tensor(max_input_len, "max_input_len")
    import pdb;pdb.set_trace()
    with record_function('opt_step0'):
        b_loc = opt_step0(Req_to_tokens, B_req_idx)
    b_loc = Req_to_tokens[B_req_idx]
    batch, head, dim = b_loc.shape[0], q.shape[1], q.shape[2]
    xq = q.view(batch, 1, head, dim).transpose(1, 2)
    for i in range(batch):
        current_arange = arange_tensor[:b_seq_len[i]]
        with record_function('opt_step1'):
            k_loc_index = opt_step1(b_seq_len[i], max_input_len, current_arange)
        k_loc = b_loc[i][k_loc_index]
        import pdb;pdb.set_trace()
        key = k[k_loc, :].view(1, b_seq_len[i], head, dim).transpose(1, 2)
        with record_function('opt_step2'):
            res = opt_step2(xq[i, :], key, dim)
        res = res.reshape(head, b_seq_len[i]).clone()
        with record_function('opt_step3'):
            res = opt_step3(res, b_start_loc[i], current_arange, out)
    import pdb;pdb.set_trace()
    return res

# @record_function('_token_attention_entrypoint')
# def _token_attention(q, k, out, Req_to_tokens, B_req_idx, b_start_loc, b_seq_len, max_input_len):
#     b_loc = Req_to_tokens[B_req_idx]
#     batch, head, dim = b_loc.shape[0], q.shape[1], q.shape[2]
#     q_device = q.device
#     xq = q.view(batch, 1, head, dim).transpose(1, 2)
#     for i in range(batch):
#         index = max_input_len - b_seq_len[i] + torch.arange(0, b_seq_len[i], device=q_device)
#         k_loc = b_loc[i][index]
#         key = k[k_loc, :].view(1, b_seq_len[i], head, dim).transpose(1, 2)
#         out_loc = b_start_loc[i] + torch.arange(0, b_seq_len[i], device=q_device)
#         tmp = torch.matmul(xq[i, :].to(torch.float32), key.transpose(2, 3).to(torch.float32)) / math.sqrt(dim)
        
#         tmp = tmp.to(torch.float16)
#         cpu_tmp = tmp.cpu()
#         cpu_out = out.cpu()
#         cpu_out[:, out_loc.cpu()] = cpu_tmp.reshape(head, b_seq_len[i].cpu())
#         out = cpu_out.cuda()
#         # out[:, out_loc] = tmp.reshape(head, b_seq_len[i])
#     return out

token_att_fwd = _token_attention
