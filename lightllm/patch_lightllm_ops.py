import torch
from torch import Tensor
import torch.nn.functional as F
import math

from typing import List, Sequence

# rotary_emb
@torch._custom_op.impl.custom_op('lightllm::rotary_emb')
def rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    ...

@rotary_emb.impl_abstract()
def lightllm_rotary_emb_abstract(x, cos, sin):
    return torch.empty_like(x)

@rotary_emb.impl(['cpu', 'cuda'])
def lightllm_rotary_emb_impl(x, cos, sin):
    seq_len, h, dim = x.shape
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    x0 = x[:, :, 0: dim // 2]
    x1 = x[:, :, dim // 2: dim]
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.cat((o0, o1), dim=-1)

@torch._custom_op.impl.custom_op('lightllm::rms_norm')
def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    ...

@rms_norm.impl_abstract()
def lightllm_rms_norm_abstract(x, weight, eps):
    return torch.empty_like(x)

@rms_norm.impl(['cpu', 'cuda'])
def lightllm_rms_norm_impl(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

@torch._custom_op.impl.custom_op('lightllm::flash_attention_inference')
def flash_attention_inference(q: Tensor, all_k: Tensor, all_v: Tensor, currnet_len: Sequence[int], max_len: int) -> Tensor:
    ...

@flash_attention_inference.impl_abstract()
def flash_attention_inference2_abstract(q: Tensor, all_k: Tensor, all_v: Tensor, currnet_len: Sequence[int], max_len: int):
    return torch.empty_like(q)

@flash_attention_inference.impl(['cpu', 'cuda'])
def flash_attention_inference2_impl(q, all_k, all_v, current_len, max_len):
    ...


@torch._custom_op.impl.custom_op('lightllm::flash_attention_inference3')
def flash_attention_inference3(q: Tensor, k_list: Sequence[Tensor], v_list: Sequence[Tensor]) -> Tensor:
    ...

@flash_attention_inference3.impl_abstract()
def flash_attention_inference3_abstract(q: Tensor, k_list: Sequence[Tensor], v_list: Sequence[Tensor]):
    # batch, head, dim = q.shape
    # kv_seq_len = batch_kv_loc.shape[0] // batch
    # k = all_k[batch_kv_loc].reshape(batch, kv_seq_len, head, dim)
    return torch.empty_like(q)

@flash_attention_inference3.impl(['cpu', 'cuda'])
def flash_attention_inference3_impl(q, k_list, v_list):
    # q: batch, head, dim
    batch = q.shape[0]
    head = q.shape[1]
    dim = q.shape[2]
    
    res = []
    compute_batch = 1
    for i in range(batch):
        kv_seq_len = k_list[i].shape[0]
        k = k_list[i].reshape(compute_batch, kv_seq_len, head, dim)
        v = v_list[i].reshape(compute_batch, kv_seq_len, head, dim)

        xq = q[i].view(compute_batch, 1, head, dim).transpose(1, 2).transpose(0, 1)   # shape: head, batch, 1, dim
        bmm_xq = xq.reshape(head * compute_batch, 1, dim)
        bmm_xk = k.transpose(1, 2).transpose(0, 1).transpose(2, 3).reshape(head * compute_batch, dim, kv_seq_len)
        
        # q @ k
        out = torch.bmm(bmm_xq, bmm_xk)
        out = out.reshape(head, compute_batch, 1, -1).reshape(head, compute_batch, -1)
        
        # softmax
        out = out.softmax(-1).reshape(head, compute_batch, 1, kv_seq_len).transpose(0, 1) # shape: batch head 1 seq_len
        xv = v.transpose(1, 2) # shape: batch head, seq_len, dim
        out = torch.bmm(out.reshape(compute_batch * head, 1, kv_seq_len), xv.reshape(compute_batch * head, kv_seq_len, dim))
        
        out = out.reshape(compute_batch, head, 1, dim).view(compute_batch, head, dim)
        res.append(out)
    res = torch.cat(res)
    return res

@torch._custom_op.impl.custom_op('lightllm::copy_with_offset')
def copy_with_offset(x: Tensor, src: Tensor, dim: int) -> Tensor:
    ...

@copy_with_offset.impl_abstract()
def copy_with_offset_abstract(x: Tensor, src: Tensor, dim: int):
    return x

@copy_with_offset.impl(['cpu', 'cuda'])
def copy_with_offset_impl(x, src, dim):
    x[dim] = src
    return x

@torch._custom_op.impl.custom_op('lightllm::copy_with_offset2')
def copy_with_offset2(x: Tensor, src: Tensor, dim: int) -> Tensor:
    ...

@copy_with_offset2.impl_abstract()
def copy_with_offset2_abstract(x: Tensor, src: Tensor, dim: int):
    return x

@copy_with_offset2.impl(['cpu', 'cuda'])
def copy_with_offset2_impl(x, src, dim):
    x[dim] = src
    return x

@torch._custom_op.impl.custom_op('lightllm::prompt_attention_inference')
def prompt_attention_inference(q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
    ...

@prompt_attention_inference.impl_abstract()
def lightllm_prompt_attention_inference_abstract(q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
    return torch.empty_like(q)

@prompt_attention_inference.impl(['cpu', 'cuda'])
def lightllm_prompt_attention_inference_impl(q, k, v, mask):
    bs = 1
    head_dim = 128
    kvnum_head = 8
    num_head = 32
    seqlen = 128
    #seqlen = seqlen.item()

    xq = q.view(bs, num_head, seqlen, head_dim)
    xk = k.view(bs, kvnum_head, seqlen, head_dim)
    xv = v.view(bs, kvnum_head, seqlen, head_dim)

    # mask = torch.tril(torch.ones(seqlen[0], seqlen[0]), diagonal=1).unsqueeze(0).unsqueeze(0)
    # mask[mask == 0.] = -100000000.0
    # mask = mask.repeat(bs, num_head, 1, 1)
    mask = mask.repeat(bs, num_head, 1, 1)

    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0)
    import numpy as np
    import pdb;pdb.set_trace()

    # 获取float32的最小值
    min_val_float32 = np.finfo(np.float16).min

    # print(min_val_float32)
    mask = mask.masked_fill(mask == 0., min_val_float32)
    mask = mask.masked_fill(mask == 1., 0.0)
    mask = mask.repeat(bs, num_head, 1, 1)
    # import pdb; pdb.set_trace()
    # mask = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool), diagonal=0)
    # mask = mask.repeat(1, 1, 1)
    # mask = torch.logical_not(mask)

    #import pdb;pdb.set_trace()
    keys = xk.float()
    values = xv.float()
    xq = xq.float()
    keys = torch.repeat_interleave(keys, num_head // kvnum_head, dim=1)
    values = torch.repeat_interleave(values, num_head // kvnum_head, dim=1)

    keys = keys.transpose(2, 3)
    #values = xv.transpose(1, 2).float()

    # print(xq.dtype, keys.dtype)
    scores = torch.matmul(xq, keys) / math.sqrt(head_dim)
    # scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    # print(scores.dtype, mask.dtype)
    scores = F.softmax(scores + mask.float(), dim=-1).type_as(xq)

    output = torch.matmul(scores, values)#.transpose(1, 2).contiguous()#.reshape(-1, num_head, head_dim)

    return output.half()

if __name__ == '__main__':
    import torch_dipu
    import torch._dynamo as dynamo
    
    torch._dynamo.config.suppress_errors = False
    
    # # rotary_emb
    # def test_rotary_emb(x, cos, sin):
    #     return torch.ops.lightllm.rotary_emb.default(x, cos, sin)
    
    # input_x = torch.randn(2, 32, 128)
    # input_cos = torch.randn(2, 64)
    # input_sin = torch.randn(2, 64)

    # aten_out = test_rotary_emb(input_x, input_cos, input_sin)
    # print(aten_out)
    # print('x.shape:', input_x.shape)
    # print('aten_out.shape:', aten_out.shape)

    # compiled_fn = torch.compile(test_rotary_emb, backend='ascendgraph', dynamic=False)

    # ascend_out = compiled_fn(input_x.cuda(), input_cos.cuda(), input_sin.cuda())
    # print(ascend_out)
    # print(ascend_out.shape)
    
    # # rms_norm
    # def ascend_rms_norm(x, weight, eps):
    #     return torch.ops.lightllm.rms_norm.default(x, weight, eps)

    # input_x = torch.randn(2, 32)
    # input_weight = torch.randn(32)
    # input_eps = 1e-3

    # aten_out = ascend_rms_norm(input_x, input_weight, input_eps)
    # print(aten_out)
    # print('x.shape:', input_x.shape)
    # print('aten_out.shape:', aten_out.shape)

    # compiled_fn = torch.compile(ascend_rms_norm, backend='ascendgraph', dynamic=False)

    # ascend_out = compiled_fn(input_x.cuda(), input_weight.cuda(), input_eps)
    # print(ascend_out)
    # print(ascend_out.shape)


    def ascend_rms_norm(x, weight, eps):
        return torch.ops.lightllm.rms_norm.default(x, weight, eps)

    def torch_rms_norm(x, weight, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


    # input_x = torch.randn(2, 32)
    # input_weight = torch.randn(32)
    input_x = torch.arange(0, 64, dtype=torch.float32).reshape(2, 32)
    input_weight = torch.arange(0, 32, dtype=torch.float32)
    input_eps = 1e-6

    torch._dynamo.config.suppress_errors = False

    # aten_out = torch_rms_norm(input_x, input_weight, input_eps)
    # print(aten_out)
    # print('aten_out.shape:', aten_out.shape)

    compiled_fn = torch.compile(ascend_rms_norm, backend='ascendgraph', dynamic=False)

    ascend_out = compiled_fn(input_x.cuda(), input_weight.cuda(), input_eps)
    print(ascend_out)
    print("ascend_out.shape:", ascend_out.shape)

    # print(aten_out - ascend_out.cpu())
