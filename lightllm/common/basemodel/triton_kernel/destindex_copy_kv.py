import torch


@torch.no_grad()
def destindex_copy_kv(k, dest_loc, out):
    out[dest_loc] = k
    return out
