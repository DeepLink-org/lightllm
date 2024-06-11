import torch

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

compiled_torch_rotary_emb = torch_rotary_emb


rotary_emb_fwd = compiled_torch_rotary_emb

def rotary_emb_v2_fwd(q, cache_k, cos, sin):
    print("should run rotary_embedding_v2 in ext")
    return None

def rotary_emb(input, cos, sin):
        x1, x2 = input.chunk(2, dim=-1)
        input_rotate = torch.cat((-x2, x1), dim=-1) 
        return input * cos + input_rotate * sin 
       

def rotary_emb_v2(query, key, cos, sin, dim):
    query = query.view(query.shape[0], -1, dim)
    key = key.view(key.shape[0], -1, dim)
    q1, q2 = query.chunk(2, dim=-1)
    query_rotate = torch.cat((-q2, q1), dim=-1) 
    query = query * cos + query_rotate * sin
    k1, k2 = key.chunk(2, dim=-1)
    key_rotate = torch.cat((-k2, k1), dim=-1) 
    key = key * cos + key_rotate * sin 
    return query.view(query.shape[0], -1), key.view(key.shape[0], -1)

def test_rotary_emb():
    dim = 128
    query = torch.randn((8, 4096), device='cuda')
    key = torch.randn((8, 4096), device='cuda')
    cos = torch.randn((8, 1, dim), device='cuda')
    sin = torch.randn((8, 1, dim), device='cuda')
    
    q2 = query.clone()
    k2 = key.clone()
    # ext.rotary_embedding_v2(query, key, torch.cat((cos,cos), dim=-1), torch.cat((sin, sin), dim=-1), dim)
    ext.rotary_embedding_v2(query, key, cos, sin, dim)

    
    q3, k3 = rotary_emb_v2(q2, k2, cos, sin, dim)

    # cos1, cos2 = cos.chunk(2, dim=-1)
    # sin1, sin2 = sin.chunk(2, dim=-1)
    # q2 = query.clone()
    # rotary_emb(q2.view(8, 32, dim), cos1, sin1, True, None)
    # k2 = key.clone()
    # rotary_emb(k2.view(8, 32, dim), cos2, sin2, True, None)
    print(query)
    print(q3)
    assert torch.allclose(query, q3)
    assert torch.allclose(key, k3)

if __name__ == "__main__":
    import torch_dipu
    import deeplink_ext.cpp_extensions as ext
    test_rotary_emb()