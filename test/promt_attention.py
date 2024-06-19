import torch
import torch_dipu

from dicp.vendor.AscendGraph import ext_ops


def load_tensor(name):
    import pickle
    with open(f'/data2/zhoushenglong/tmp/{name}.pkl', 'rb') as f:
            x = pickle.load(f)
    if isinstance(x, torch.Tensor):
            return x.cuda()
    return x

def equal(a, b):
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, torch.Tensor)

    res = torch.allclose(a, b, rtol=1e-2, atol=1e-2)
    
    return res

def ascend_prompt_attention_kernel(q, k, v, seqlen, num_head , head_dim, numKeyValueHeads):
    return torch.ops.lightllm.prompt_attention_inference.default(q, k, v, seqlen, num_head, head_dim, numKeyValueHeads)

opt_ascend_prompt_attention_inference = torch.compile(ascend_prompt_attention_kernel, backend='ascendgraph', dynamic=False)


q = load_tensor("q_test_compile")
cache_k = load_tensor("cache_k_test_compile")
cache_v = load_tensor("cache_v_test_compile")
seqlen = load_tensor("seqlen_test_compile")

q_t = load_tensor("q_test_op")
cache_k_t = load_tensor("cache_k_test_op")
cache_v_t = load_tensor("cache_v_test_op")

o = load_tensor("o_op")
o_t = load_tensor("out_compile_t")

# q = torch.randn(4, 16, 128, dtype=torch.float16).cuda()
# cache_k = torch.randn(4, 2, 128, dtype=torch.float16).cuda()
# cache_v = torch.randn(4, 2, 128, dtype=torch.float16).cuda()

batch = 1
num_head = q.shape[1]
head_dim = q.shape[2]

assert cache_k.shape[1] == cache_v.shape[1]
numKeyValueHeads = cache_k.shape[1]

res = opt_ascend_prompt_attention_inference(q.view(batch, -1, num_head, head_dim), 
                                            cache_k.view(batch, -1, numKeyValueHeads, head_dim), 
                                            cache_v.view(batch, -1, numKeyValueHeads, head_dim), 
                                            seqlen, num_head, head_dim, numKeyValueHeads)

res = res.reshape(-1, num_head * head_dim)

import pdb; pdb.set_trace()

# res_t = ascend_prompt_attention_kernel(q.view(batch, -1, num_head * head_dim), 
#                                        cache_k.view(batch, -1, numKeyValueHeads * head_dim), 
#                                        cache_v.view(batch, -1, numKeyValueHeads * head_dim), 
#                                        seqlen, num_head, head_dim, numKeyValueHeads)

# res_t = res_t.reshape(-1, num_head * head_dim)

# import pdb; pdb.set_trace()

