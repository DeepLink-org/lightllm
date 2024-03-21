
import math

import torch
import torch_dipu
import torch._dynamo as dynamo
import torch.nn.functional as F

from torch import Tensor

torch._dynamo.config.suppress_errors = False

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

@torch._custom_op.impl.custom_op('lightllm::prompt_attention_inference')
def prompt_attention_inference(q: Tensor, k: Tensor, v: Tensor, num_head: int, seqlen: Tensor, mask: Tensor) -> Tensor:
    ...

@prompt_attention_inference.impl_abstract()
def lightllm_prompt_attention_inference_abstract(q: Tensor, k: Tensor, v: Tensor, num_head: int, seqlen: Tensor,  mask: Tensor):
    return torch.empty_like(q)

@prompt_attention_inference.impl(['cpu', 'cuda'])
def lightllm_prompt_attention_inference_impl(q, k, v, num_head, seqlen, mask):
    bs = 1
    head_dim = 128
    seqlen = seqlen.item()

    xq = q.view(bs, seqlen, num_head, head_dim)
    xk = k.view(bs, seqlen, num_head, head_dim)
    xv = v.view(bs, seqlen, num_head, head_dim)

    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0).unsqueeze(0).unsqueeze(0)
    # mask[mask == 0.] = -100000000.0
    mask = mask.masked_fill(mask == 0., -100000000.0)
    mask = mask.masked_fill(mask == 1., 0.0)
    mask = mask.repeat(bs, num_head, 1, 1)

    keys = xk
    values = xv
    xq = xq.transpose(1, 2).float()
    keys = keys.transpose(1, 2).float()
    values = values.transpose(1, 2).float()

    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
    # scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    scores = F.softmax(scores.float() + mask, dim=-1).type_as(xq)

    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, head_dim)

    return output


if __name__ == '__main__':    
    torch._dynamo.config.suppress_errors = False
    
    # rotary_emb
    def test_rotary_emb(x, cos, sin):
        return torch.ops.lightllm.rotary_emb.default(x, cos, sin)
    
    input_x = torch.randn(2, 32, 128)
    input_cos = torch.randn(2, 64)
    input_sin = torch.randn(2, 64)

    aten_out = test_rotary_emb(input_x, input_cos, input_sin)
    # print(aten_out)
    # print('x.shape:', input_x.shape)
    # print('aten_out.shape:', aten_out.shape)

    compiled_fn = torch.compile(test_rotary_emb, backend='ascendgraph', dynamic=False)

    ascend_out = compiled_fn(input_x.cuda(), input_cos.cuda(), input_sin.cuda())
    # print(ascend_out)
    # print(ascend_out.shape)

    print(aten_out - ascend_out.cpu(), flush=True)
    
    # rms_norm
    def ascend_rms_norm(x, weight, eps):
        return torch.ops.lightllm.rms_norm.default(x, weight, eps)

    input_x = torch.randn(2, 32)
    input_weight = torch.randn(32)
    input_eps = 1e-3

    aten_out = ascend_rms_norm(input_x, input_weight, input_eps)
    # print(aten_out)
    # print('x.shape:', input_x.shape)
    # print('aten_out.shape:', aten_out.shape)

    compiled_fn = torch.compile(ascend_rms_norm, backend='ascendgraph', dynamic=False)

    ascend_out = compiled_fn(input_x.cuda(), input_weight.cuda(), input_eps)
    # print(ascend_out)
    # print(ascend_out.shape)

    print(aten_out - ascend_out.cpu(), flush=True)

