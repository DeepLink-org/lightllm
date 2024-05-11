import math
import torch
from torch.profiler import record_function
import torch_dipu
import deeplink_ext.cpp_extensions as ext
from contextlib import nullcontext

# @triton.jit
# def _fwd_kernel_token_att1(
#     Q, K, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
#     Att_Out,
#     stride_req_to_tokens_b, stride_req_to_tokens_s,
#     stride_qbs, stride_qh, stride_qd,
#     stride_kbs, stride_kh, stride_kd,
#     att_stride_h, att_stride_bs,
#     kv_group_num,
#     BLOCK_DMODEL: tl.constexpr,
#     BLOCK_N: tl.constexpr
# ):
#     cur_batch = tl.program_id(0)
#     cur_head = tl.program_id(1)
#     start_n = tl.program_id(2)
    
#     cur_kv_head = cur_head // kv_group_num

#     offs_d = tl.arange(0, BLOCK_DMODEL)
#     cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
#     cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
#     cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

#     cur_batch_start_index = 0
#     cur_batch_end_index = cur_batch_seq_len

#     off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

#     offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

#     block_stard_index = start_n * BLOCK_N
#     block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

#     for start_mark in range(0, block_mask, 1):
#         q = tl.load(Q + off_q + start_mark)
#         offs_n_new = cur_batch_start_index + offs_n
#         k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * offs_n_new, 
#                         mask=offs_n_new < cur_batch_end_index, other=0)
#         off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
#         k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
#         att_value = tl.sum(q[None, :] * k, 1)
#         att_value *= sm_scale
#         off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
#         tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
#     return


# @torch.no_grad()
# def token_att_fwd(q, k, att_out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, max_len_in_batch):
#     BLOCK = 32
#     # shape constraints
#     Lq, Lk = q.shape[-1], k.shape[-1]
#     assert Lq == Lk
#     assert Lk in {16, 32, 64, 128}
#     sm_scale = 1.0 / (Lk ** 0.5)

#     batch, head_num = B_req_idx.shape[0], q.shape[1]

#     grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK))
#     kv_group_num = q.shape[1] // k.shape[1]
    
#     if kv_group_num == 1:
#         num_warps = 4
#     else:
#         num_warps = 2

#     _fwd_kernel_token_att1[grid](
#         q, k, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
#         att_out,
#         Req_to_tokens.stride(0), Req_to_tokens.stride(1),
#         q.stride(0), q.stride(1), q.stride(2),
#         k.stride(0), k.stride(1), k.stride(2),
#         att_out.stride(0), att_out.stride(1),
#         kv_group_num=kv_group_num,
#         BLOCK_DMODEL=Lk,
#         BLOCK_N=BLOCK,
#         num_warps=num_warps,
#         num_stages=1,
#     )
#     return


# @triton.jit
# def _fwd_kernel_token_att1_int8(
#     Q, K, K_scale, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
#     Att_Out,
#     stride_req_to_tokens_b, stride_req_to_tokens_s,
#     stride_qbs, stride_qh, stride_qd,
#     stride_kbs, stride_kh, stride_kd,
#     stride_ksbs, stride_ksh, stride_ksd,
#     att_stride_h, att_stride_bs,
#     BLOCK_DMODEL: tl.constexpr,
#     BLOCK_N: tl.constexpr
# ):
#     cur_batch = tl.program_id(0)
#     cur_head = tl.program_id(1)
#     start_n = tl.program_id(2)

#     offs_d = tl.arange(0, BLOCK_DMODEL)
#     cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
#     cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
#     cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

#     cur_batch_start_index = 0
#     cur_batch_end_index = cur_batch_seq_len

#     off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

#     offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

#     block_stard_index = start_n * BLOCK_N
#     block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

#     for start_mark in range(0, block_mask, 1):
#         q = tl.load(Q + off_q + start_mark)
#         offs_n_new = cur_batch_start_index + offs_n
#         k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
#         off_k = k_loc[:, None] * stride_kbs + cur_head * stride_kh + offs_d[None, :] * stride_kd
#         k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
#         off_ks = k_loc[:, None] * stride_ksbs + cur_head * stride_ksh
#         k_scale = tl.load(K_scale + off_ks, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
#         att_value = tl.sum(q[None, :] * k * k_scale, 1)
#         att_value *= sm_scale
#         off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
#         tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
#     return


# @torch.no_grad()
# def token_att_fwd_int8k(q, k, k_scale, att_out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, max_input_len):
#     BLOCK = 32
#     # shape constraints
#     Lq, Lk = q.shape[-1], k.shape[-1]
#     assert Lq == Lk
#     assert Lk in {16, 32, 64, 128}
#     sm_scale = 1.0 / (Lk ** 0.5)

#     batch, head_num = B_req_idx.shape[0], q.shape[1]

#     grid = (batch, head_num, triton.cdiv(max_input_len, BLOCK))

#     num_warps = 4 if Lk <= 64 else 8
#     num_warps = 2

#     _fwd_kernel_token_att1_int8[grid](
#         q, k, k_scale, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
#         att_out,
#         Req_to_tokens.stride(0), Req_to_tokens.stride(1),
#         q.stride(0), q.stride(1), q.stride(2),
#         k.stride(0), k.stride(1), k.stride(2),
#         k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
#         att_out.stride(0), att_out.stride(1),
#         BLOCK_DMODEL=Lk,
#         BLOCK_N=BLOCK,
#         num_warps=num_warps,
#         num_stages=1,
#     )
#     return




def step0(Req_to_tokens, B_req_idx):
    b_loc = Req_to_tokens[B_req_idx]
    return b_loc
opt_step0 = step0 #torch.compile(step0, backend='ascendgraph', dynamic=False)

def step1(b_seq_len, max_input_len, current_arange):
    k_loc_index = max_input_len - b_seq_len + current_arange
    return k_loc_index
opt_step1 = step1 #torch.compile(step1, backend='ascendgraph', dynamic=False)

def step2(xq, key, dim):
    return torch.matmul(xq, key.transpose(2, 3)) / math.sqrt(dim)
opt_step2 = step2 #torch.compile(step2, backend='ascendgraph', dynamic=False)

def step3(input, b_start_loc, current_arange, out):
    out_loc = b_start_loc + current_arange
    out[:, out_loc] = input
    return out
opt_step3 = step3 #torch.compile(step3, backend='ascendgraph', dynamic=False)

@record_function('_token_attention_entrypoint')
def _token_attention(q, k, out, Req_to_tokens, B_req_idx, b_start_loc, b_seq_len, max_input_len):
    arange_tensor = torch.arange(0, max_input_len).cuda()
    with record_function('opt_step0'):
        b_loc = opt_step0(Req_to_tokens, B_req_idx)
    b_loc = Req_to_tokens[B_req_idx]
    batch, head, dim = b_loc.shape[0], q.shape[1], q.shape[2]
    kv_num_head = k.shape[1]
    xq = q.view(batch, 1, head, dim).transpose(1, 2)
    for i in range(batch):
        current_arange = arange_tensor[:b_seq_len[i]]
        with record_function('opt_step1'):
            k_loc_index = opt_step1(b_seq_len[i], max_input_len, current_arange)
        k_loc = b_loc[i][k_loc_index]
        # import pdb; pdb.set_trace()
        key = k[k_loc, :].view(1, b_seq_len[i], kv_num_head, dim).transpose(1, 2)
        with record_function('opt_step2'):
            res = opt_step2(xq[i, :], key, dim)
        res = res.reshape(head, b_seq_len[i]).clone()
        with record_function('opt_step3'):
            res = opt_step3(res, b_start_loc[i], current_arange, out)
    return res

token_att_fwd = _token_attention

def token_decode_attention_fwd(q, k, v, out, req_to_tokens, b_req_idx, b_start_loc, b_seq_len, max_len_in_batch, other_kv_index):
    print("running token_decode_attention_fwd")
    return

def dump_tensor(x, name):
    import pickle
    with open(f'/data02/zhoushenglong/tmp/{name}.pkl', 'wb') as f:
        if isinstance(x, torch.Tensor):
            pickle.dump(x.cpu(), f)
        else:
            pickle.dump(x, f)

def torch_token_attention(q, k, v, out, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index):
    # q: BSH
    print("************* debug info ***************************", flush=True)
    print("b_loc", b_loc, flush=True)
    print(max_input_len, b_seq_len, torch.arange(0, b_seq_len[0], device="cuda:0", dtype=torch.int32), flush=True)
    # dump_tensor(key, "key_test1")
    print("************* debug info ***************************", flush=True)
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
        b_seq_len = torch.tensor([1025], dtype=torch.int32).cuda()
    elif q.shape[0] == 2:
        req_to_token_indexs = torch.zeros((2, 512), dtype=torch.int32).cuda()
        req_to_token_indexs[0, 0:129] = torch.arange(0, 129, 1, dtype=torch.int32).cuda()
        req_to_token_indexs[1, 0:129] = torch.arange(128, 257, 1, dtype=torch.int32).cuda()
        b_req_idx = torch.tensor([0, 1], dtype=torch.int32).cuda()
        b_start_loc = torch.tensor([0, 129], dtype=torch.int32).cuda()
        b_seq_len = torch.tensor([129, 129], dtype=torch.int32).cuda()
    max_len_in_batch = 1025
    other_kv_index = 0
    o_torch = torch.empty_like(q)

    torch_token_attention(q, k, v, o_torch, req_to_token_indexs[b_req_idx],
                          b_start_loc, b_seq_len, max_len_in_batch, other_kv_index)
    print(o_torch)

    o_ext = torch.empty_like(q)
    ext_token_attention(q, k, v, o_ext, req_to_token_indexs[b_req_idx],
                          b_start_loc, b_seq_len, max_len_in_batch, other_kv_index)
    print(o_ext)
    print(torch.allclose(o_torch.cpu(), o_ext.cpu(), rtol=1e-2, atol=1e-2))

if __name__ == "__main__":
    # test_incre_flash_attn()
    test_token_attention()
