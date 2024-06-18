import torch


@torch.no_grad()
def destindex_copy_kv(k, dest_loc, out):
    # out[dest_loc] = k
    out = torch.ops.lightllm.copy_with_index.default(out, k, dest_loc)
    return out
