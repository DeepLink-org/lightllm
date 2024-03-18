import torch
from torch.profiler import record_function

def torch_rotary_emb(x, cos, sin):
    seq_len, h, dim = x.shape
    x0 = x[:, :, 0: dim // 2]
    x1 = x[:, :, dim // 2: dim]
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    x.copy_(torch.cat((o0, o1), dim=-1))
    return

compiled_torch_rotary_emb = torch.compile(torch_rotary_emb, backend='ascendgraph', dynamic=False)


rotary_emb_fwd = compiled_torch_rotary_emb
