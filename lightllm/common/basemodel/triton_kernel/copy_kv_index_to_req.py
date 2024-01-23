import torch


@torch.no_grad()
def copy_kv_index_to_req(req_to_token_indexs, b_req_idx, b_seq_len, memindex):
    seq_len = b_seq_len.shape[0]
    assert b_seq_len.shape[0] == memindex.shape[0] and b_req_idx.shape[0] == b_seq_len.shape[0]
    tmp_memindex = memindex.to(torch.int32)
    req_to_token_indexs[b_req_idx, b_seq_len - 1] = tmp_memindex
    return
