import numpy
import torch
from torch import Tensor
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.paging.block import _div_up
from lightllm.common.paging.block_manager.base_block_manager import BaseBlockManager
from lightllm.common.paging.block_manager.default_block_manager import DefaultBlockManager
from lightllm.common.paging.request import Request
from lightllm.common.req_manager import ReqManager
from lightllm.utils.log_utils import init_logger
import lightllm.common.basemodel.triton_kernel.destindex_copy_kv


logger = init_logger(__name__)

    
class PagingRequestManager(ReqManager):
    BLOCK_SIZE = 128
    def __init__(self, max_request_num, max_sequence_length, mem_manager):
        self.req_state = torch.zeros((max_request_num,), dtype=torch.bool, device="cuda")
        self.req_to_token_indexs = torch.zeros((max_request_num, max_sequence_length), dtype=torch.int32, device="cuda")
        self.mem_index_offset = torch.arange(0, max_request_num*max_sequence_length, max_sequence_length, dtype=torch.int32, device="cuda")
        self.can_use_req_size = max_request_num
        self.mem_manager: MemoryManager = mem_manager
        self.req_map: dict[int, Request] = {}
        self.num_blocks = _div_up(self.mem_manager.size, PagingRequestManager.BLOCK_SIZE)
        self.block_manager: BaseBlockManager = DefaultBlockManager(self.num_blocks)


    @torch.no_grad()
    def alloc_page(self, req_idx: torch.Tensor, seq_len: list):
        assert req_idx.shape[0] == len(seq_len)
        for idx, length in zip(req_idx.tolist(), seq_len):
            if idx not in self.req_map:
                self.req_map[idx] = Request(length, PagingRequestManager.BLOCK_SIZE)
            req = self.req_map[idx]
            req.update_seq_len(length)
            if False == self.block_manager.can_allocate(req):
                raise Exception(f"can not allocate block memory, req_idx:{req_idx}, seqLen:{seq_len}")
            self.block_manager.allocate(req)


    def get_block_table(self, req_idx: int):
        assert req_idx in self.req_map
        return self.block_manager.get_block_table(self.req_map[req_idx])
    
    def get_batched_block_table(self, req_idx: Tensor):
        batch = req_idx.shape[0]
        table = [ self.get_block_table(int(req_idx[i])) for i in range(batch) ]
        length = [ len(t) for t in table]
        max_len = max(length)
        padding_table = []
        for t in table:
            if len(t) < max_len:
                tmp = [-1 for i in range(max_len - len(t))]
                padding_table.append(numpy.append(t, tmp))
            else:
                padding_table.append(t)
        return torch.from_numpy(numpy.array(padding_table)).cuda().to(torch.int32)
        

    def fill_kv_cache(self, layer_num: int, k: Tensor, v: Tensor, infer_state:InferStateInfo):
        assert k.shape[0] == v.shape[0]
        if infer_state.is_prefill:
            self.fill_kv_cache_prefill(layer_num, k, v, infer_state)
        else:
            self.fill_kv_cache_decode(layer_num, k, v, infer_state)

    def fill_kv_cache_decode(self, layer_num: int, k: Tensor, v: Tensor, infer_state:InferStateInfo):
        lightllm.common.basemodel.triton_kernel.destindex_copy_kv.destindex_copy_kv(k, infer_state.kv_start_indices, self.mem_manager.key_buffer[layer_num])
        lightllm.common.basemodel.triton_kernel.destindex_copy_kv.destindex_copy_kv(v, infer_state.kv_start_indices, self.mem_manager.value_buffer[layer_num])
        

    def fill_kv_cache_prefill(self, layer_num: int, k: Tensor, v: Tensor, infer_state:InferStateInfo):
        if k.shape[0] == infer_state.kv_start_indices.shape[0]:
            nopad_k, nopad_v = k, v
        else:
            nopad_k, nopad_v = k[infer_state.kv_start_indices], v[infer_state.kv_start_indices]
        lightllm.common.basemodel.triton_kernel.destindex_copy_kv.destindex_copy_kv(nopad_k, infer_state.block_indices, self.mem_manager.key_buffer[layer_num])
        lightllm.common.basemodel.triton_kernel.destindex_copy_kv.destindex_copy_kv(nopad_v, infer_state.block_indices, self.mem_manager.value_buffer[layer_num])

    def alloc(self, need_size):
        if need_size > self.can_use_req_size:
            logger.error(f'Insufficient requested capacity, remaining {self.can_use_req_size}')
            return None
        select_index = torch.nonzero(self.req_state==0).reshape(-1)[:need_size]
        self.req_state[select_index] = 1
        self.can_use_req_size -= len(select_index)
        return select_index
    
    def free(self, free_req_index, free_token_index):
        self.can_use_req_size += len(free_req_index)
        self.req_state[free_req_index] = 0
        if self.can_use_req_size == len(self.req_state):
            logger.debug(f"freed all request size {self.can_use_req_size}")
        self.free_blocks(free_req_index)

    def free_blocks(self, free_req_index):
        for idx in free_req_index:
            if idx in self.req_map:
                self.block_manager.free(self.req_map[idx])
                self.req_map.pop(idx)
    
    def free_req(self, free_req_index):
        self.can_use_req_size +=1
        self.req_state[free_req_index] = 0
        self.free_blocks(free_req_index)
        return
    
    def free_token(self, free_token_index):
        pass

    def free_all(self):
        self.can_use_req_size = len(self.req_state)
        self.req_state[:] = 0
    
    
