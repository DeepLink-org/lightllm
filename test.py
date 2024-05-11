import torch
import torch_dipu

import pickle

def load_tensor(name):
    with open(f'/data02/zhoushenglong/tmp/{name}.pkl', 'rb') as f:
        x = pickle.load(f)
    if isinstance(x, torch.Tensor):
        # if x.dtype == torch.float16:
        #     x = x.to(torch.float32)
        return x.cuda()
    return x

def compare(a, b):
    assert(isinstance(a, torch.Tensor))
    assert(isinstance(b, torch.Tensor))

    flag = torch.allclose(a.cpu(), b.cpu(), rtol=1e-02, atol=1e-01, equal_nan=True)

    return flag

# dims = [ i for i in range(0, 256)]
# print(dims)




q_1 = load_tensor("q_1")
k_1 = load_tensor("k_1")
v_1 = load_tensor("v_1")
b_req_tokens_1 = load_tensor("b_req_tokens_1")
mem_index_1 = load_tensor("mem_index_1")
b_start_loc_1 = load_tensor("b_start_loc_1")
b_req_idx_1 = load_tensor("b_req_idx_1")
b_seq_len_1 = load_tensor("b_seq_len_1")
max_len_in_batch_1 = load_tensor("max_len_in_batch_1")
other_kv_index_1 = load_tensor("other_kv_index_1")
o_tensor_1 = load_tensor("o_tensor_1")


q_8 = load_tensor("q_8")
k_8 = load_tensor("k_8")
v_8 = load_tensor("v_8")
b_req_tokens_8 = load_tensor("b_req_tokens_8")
mem_index_8 = load_tensor("mem_index_8")
b_start_loc_8 = load_tensor("b_start_loc_8")
b_req_idx_8 = load_tensor("b_req_idx_8")
b_seq_len_8 = load_tensor("b_seq_len_8")
max_len_in_batch_8 = load_tensor("max_len_in_batch_8")
other_kv_index_8 = load_tensor("other_kv_index_8")
o_tensor_8 = load_tensor("o_tensor_8")

import pdb; pdb.set_trace()

# print(compare(predict_logics_v100, predict_logics_huawei))
