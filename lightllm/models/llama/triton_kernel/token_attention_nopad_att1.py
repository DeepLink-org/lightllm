import math
import torch
from torch import Tensor
from torch.profiler import record_function


from contextlib import nullcontext

# token_att_fwd = _token_attention

def token_decode_attention_fwd(q, k, v, out, req_to_tokens, b_req_idx, b_start_loc, b_seq_len, max_len_in_batch, other_kv_index):
    print("running token_decode_attention_fwd")
    return

def paged_token_attention(q, k_cache, v_cache, out, b_seq_len, block_table, block_size):
    raise Exception("should running paged_token_attention in ext")

def matmul_all_reduce(out, x1, x2, bias, group):
    print("running matmul_all_reduce")
    return

def ext_paged_attention(q: Tensor, k_cache: Tensor, v_cache: Tensor, current_lens, block_table: Tensor, block_size: int):
    numKeyValueHeads = k_cache.shape[1]
    assert k_cache.shape[1] == v_cache.shape[1]
    batch, head, dim = q.shape
    kv_cache_len = k_cache.shape[0]
    q = q.reshape(batch, head*dim).unsqueeze(1)
    k_cache = k_cache.reshape(kv_cache_len, numKeyValueHeads*dim).unsqueeze(0)
    v_cache = v_cache.reshape(kv_cache_len, numKeyValueHeads*dim).unsqueeze(0)
    # current_lens = b_seq_len.cpu().numpy().tolist()
    out = torch.empty_like(q)
    
    ext.paged_attention(out, q, k_cache, v_cache,
                        None, None, 
                        current_lens, block_table, head, numKeyValueHeads,
                        1.0 / math.sqrt(dim), "BSH", block_size, 0, 
                        None, None, None, None, None, None, None, None
                        )
    return out.squeeze().reshape((batch, head, dim))


def torch_token_attention(q, k, v, out, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index):
    # q: BSH
    batch, head, dim = b_loc.shape[0], q.shape[1], q.shape[2]
    q_device = q.device
    xq = q.view(batch, 1, head, dim).transpose(1, 2)
    for i in range(batch):
        # token_attention
        k_loc = b_loc[i][max_input_len - b_seq_len[i] + torch.arange(0, b_seq_len[i], device=q_device, dtype=torch.int32)]
        key = k[k_loc, :].view(1, b_seq_len[i], head, dim).transpose(1, 2)
        # out_loc = b_start_loc[i] + torch.arange(0, b_seq_len[i], device=q_device)
        # out[:, out_loc] = (torch.matmul(xq[i, :], key.transpose(2, 3)) / math.sqrt(dim)).reshape(head, b_seq_len[i])
        logics = (torch.matmul(xq[i, :], key.transpose(2, 3)) / math.sqrt(dim)).reshape(head, b_seq_len[i])

        # token_softmax_reducev
        v_loc = b_loc[i][max_input_len - b_seq_len[i] + torch.arange(0, b_seq_len[i], device=logics.device, dtype=torch.int32)]
        # P = logics[:, b_start_loc[i]:b_start_loc[i] + b_seq_len[i]].softmax(-1).reshape(1, head,  1, b_seq_len[i])
        P = logics.softmax(-1).reshape(1, head,  1, b_seq_len[i])
        V = v[v_loc, :].view(1, b_seq_len[i], head, dim).transpose(1, 2)
        out[i, :] = torch.matmul(P, V).view(head, dim)
    return

def ext_token_attention(q, k, v, out, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index):
    if q.shape[0] >= 2:
    # if True:
        print("use ext.token_decode_attention_inference")
        ext.token_decode_attention_inference(q, k, v, out, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index)
    else:
        print("use ext.token_decode_attention_inference_batch_one")
        ext.token_decode_attention_inference_batch_one(q, k, v, out, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index)
    return


def ext_incre_flash_attn(q: torch.Tensor, k, v, head=None):
    if q.dim() == 4:
        layout = "BNSD"
        head = q.shape[1]
    else:
        layout = "BSH"
        assert head != None
    out = torch.empty_like(q)
    ext.incre_flash_attention(q, k, v, out, head, layout)
    return out

def trans_BNSD2BSH(self, tensor: torch.Tensor):
    tensor = torch.transpose(tensor, 1, 2)
    tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
    return tensor

def torch_incre_flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head=None):
    if q.dim() == 4:
        # layout BNSD
        head = q.shape[1]
        embed_dim = q.shape[3]
        hidden_size = head * embed_dim
        query = q
        key = k
        value = v
    else:
        layout = "BSH"
        assert head != None
        hidden_size = q.shape[2]
        embed_dim = hidden_size / head
        query = trans_BNSD2BSH(q)
        key = trans_BNSD2BSH(k)
        value = trans_BNSD2BSH(v)

    attn_weights1 = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(embed_dim)
    # attn_weights1 = torch.max(attn_weights1, torch.full(
    #     (1, 1), torch.finfo(attn_weights1.dtype).min, device=attn_weights1.device))
    attn_weights1 = torch.nn.functional.softmax(attn_weights1.float(), dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output1 = torch.matmul(attn_weights1, value)
    # attn_output1 = attn_output1.transpose(1, 2)
    # attn_output1 = attn_output1.reshape(1, 1, hidden_size)  # IFA (1, 1, 4096)
    return attn_output1


def test_incre_flash_attn():
    # BNSD
    q = torch.randn(1, 32, 1, 128, dtype=torch.float16).cuda()
    k = torch.randn(1, 32, 128, 128, dtype=torch.float16).cuda()
    v = torch.randn(1, 32, 128, 128, dtype=torch.float16).cuda()

    context = torch_dipu.profiler.NativeProfile("./incre_flash_attn_v2", False)
    context = nullcontext()

    repeat_num = 1
    with context:
        torch.cuda.synchronize()
        for i in range(repeat_num):
            with record_function(f"ext_incre_flash_attn_attempt_{i}"):
                output1 = ext_incre_flash_attn(q, k, v)
                torch.cuda.synchronize()

        for i in range(repeat_num):
            with record_function(f"torch_incre_flash_attn_attempt_{i}"):
                output2 = torch_incre_flash_attn(q, k, v)
                torch.cuda.synchronize()
    print(output1.cpu())
    print(output2.cpu())
    print(torch.allclose(output1.cpu(), output2.cpu(), rtol=1e-2, atol=1e-2))

def test_token_attention():
    # BND S=1
    q = torch.randn(1, 32, 128, dtype=torch.float16).cuda()
    # SND
    k = torch.randn(1026, 32, 128, dtype=torch.float16).cuda()
    v = torch.randn(1026, 32, 128, dtype=torch.float16).cuda()

    if q.shape[0] == 1:
        req_to_token_indexs = torch.arange(1026, dtype=torch.int32).reshape(1, 1026).cuda()
        b_req_idx = torch.tensor([0], dtype=torch.int32).cuda()
        b_start_loc = torch.tensor([0], dtype=torch.int32).cuda()
        b_seq_len = torch.tensor([150], dtype=torch.int32).cuda()
    elif q.shape[0] == 2:
        req_to_token_indexs = torch.zeros((2, 512), dtype=torch.int32).cuda()
        req_to_token_indexs[0, 0:127] = torch.arange(0, 127, 1, dtype=torch.int32).cuda()
        req_to_token_indexs[1, 0:127] = torch.arange(128, 255, 1, dtype=torch.int32).cuda()
        b_req_idx = torch.tensor([0, 1], dtype=torch.int32).cuda()
        b_start_loc = torch.tensor([0, 128], dtype=torch.int32).cuda()
        b_seq_len = torch.tensor([127, 127], dtype=torch.int32).cuda()
    max_len_in_batch = 150
    other_kv_index = 0
    o_torch = torch.empty_like(q)

    ext.token_decode_attention_inference_batch_one(q, k, v, o_torch, req_to_token_indexs[b_req_idx],
                          b_start_loc, b_seq_len, max_len_in_batch, other_kv_index)
    # print(o_torch)

    # o_ext = torch.empty_like(q)
    # ext_token_attention(q, k, v, o_ext, req_to_token_indexs[b_req_idx],
    #                       b_start_loc, b_seq_len, max_len_in_batch, other_kv_index)
    # print(o_ext)
    # assert torch.allclose(o_torch.cpu(), o_ext.cpu(), rtol=1e-2, atol=1e-2)

    # max_len_in_batch = 1025
    current_lens, blk_size = [max_len_in_batch], 128
    block_table = torch.tensor([
    # [i for i in range(max_seq_length // block_size)]
    [0,1],
    # [1],
    # [3,5,7,],
    # [1,2,8,]
    ], dtype=torch.int32, device="cuda")
    o_ext = ext_paged_attention(q, k, v,current_lens ,block_table,blk_size )
    # print(o_ext)
    print(torch.allclose(o_torch.cpu(), o_ext.cpu(), rtol=1e-2, atol=1e-2))

if __name__ == "__main__":
    import torch_dipu
    import deeplink_ext.cpp_extensions as ext
    # test_incre_flash_attn()
    for i in range(20):
        test_token_attention()
