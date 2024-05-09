import torch


@torch.no_grad()
def destindex_copy_kv(k, dest_loc, out):
    out[dest_loc] = k
    return out


def test1():
    import time

    B, N_CTX, H, D = 32, 1024, 12, 128
    dest = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    src = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    dest_loc = torch.arange(0, B * N_CTX, dtype=torch.int32, device="cuda")

    for _ in range(10):
        destindex_copy_kv(src, dest_loc, dest)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(1000):
        destindex_copy_kv(src, dest_loc, dest)
    torch.cuda.synchronize()
    t2 = time.time()

    print("Time cost ", t2 - t1)
    print("max ", torch.max(torch.abs(dest - src)))
    print("mean ", torch.mean(torch.abs(dest - src)))
    assert torch.allclose(src, dest, atol=1e-2, rtol=0)


def test2():
    import time

    B, N_CTX, H, D = 32, 1024, 12, 128
    src = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    dest_loc = torch.arange(0, B * N_CTX, dtype=torch.int32).cuda()
    value_dest = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda().to(torch.int8)
    scale_dest = torch.randn((B * N_CTX, H, 1), dtype=torch.float16).cuda()

    for _ in range(10):
        destindex_copy_quantize_kv(src, dest_loc, value_dest, scale_dest)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(1000):
        destindex_copy_quantize_kv(src, dest_loc, value_dest, scale_dest)
    torch.cuda.synchronize()
    t2 = time.time()

    print("Time cost ", t2 - t1)
    print("max ", torch.max(torch.abs(value_dest * scale_dest - src)))
    print("mean ", torch.mean(torch.abs(value_dest * scale_dest - src)))
    cos = torch.nn.CosineSimilarity(0)
    print("cos ", cos(src.flatten().to(torch.float32), (value_dest * scale_dest).flatten().to(torch.float32)))


if __name__ == '__main__':
    test1()
    test2()
