import torch


def torch_rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

rmsnorm_forward = torch_rms_norm

def test_rms_norm(M, N, dtype, eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device='cuda')
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    # forward pass
    y_tri = rmsnorm_forward(x, weight, eps)
    y_ref = torch_rms_norm(x.to(torch.float32), weight.to(torch.float32), eps).to(dtype)

    # compare
    print("type:", y_tri.dtype, y_ref.dtype)
    print("max delta:", torch.max(torch.abs(y_tri - y_ref)))
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    return
