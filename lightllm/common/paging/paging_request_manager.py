import numpy
import torch
from torch import Tensor
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.paging.block import _div_up
from lightllm.common.paging.block_manager.base_block_manager import BaseBlockManager
from lightllm.common.paging.block_manager.default_block_manager import DefaultBlockManager
from lightllm.common.paging.request import Request
from lightllm.common.req_manager import ReqManager
from lightllm.utils.log_utils import init_logger



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
        # print(f"table:{table}, max_len:{max_len}, length:{length}")
        return torch.from_numpy(numpy.array(padding_table)).cuda().to(torch.int32)
        

    def fill_kv_cache(self, req_idx: Tensor, b_start_loc:Tensor, b_seq_len:list, layer_num: int, k: Tensor, v: Tensor):
        assert k.shape[0] == v.shape[0]
        batch = b_start_loc.shape[0]
        if batch == k.shape[0]:
            self.fill_kv_cache_decode(req_idx, b_seq_len, layer_num, k, v)
        else:
            self.fill_kv_cache_prefill(req_idx, b_start_loc, b_seq_len, layer_num, k, v)

    def fill_kv_cache_decode(self, req_idx: Tensor, b_seq_len:list, layer_num: int, k: Tensor, v: Tensor):
        assert k.shape[0] == len(b_seq_len)
        batch = k.shape[0]
        for b_idx in range(batch):
            seq_len = b_seq_len[b_idx]
            req = self.req_map[int(req_idx[b_idx])]
            block_table = self.block_manager.get_block_table(req)
            
            block_idx = block_table[req.num_blocks() - 1]
            last_block_offset = (seq_len - 1) % PagingRequestManager.BLOCK_SIZE
            cache_start = block_idx * PagingRequestManager.BLOCK_SIZE + last_block_offset
            self.mem_manager.key_buffer[layer_num][cache_start] = k[b_idx]
            self.mem_manager.value_buffer[layer_num][cache_start] = v[b_idx]
            # if layer_num == 2 and seq_len >= 125 and seq_len <= 130:
            #     print(f"block_table:{self.get_batched_block_table(req_idx)}, block_idx:{block_idx}, last_block_offset:{last_block_offset}")
            #     print(f"seqlen:{seq_len}, k:{k.view(-1)}")
        

    def fill_kv_cache_prefill(self, req_idx: Tensor, b_start_loc:Tensor, b_seq_len:list, layer_num: int, k: Tensor, v: Tensor):
        batch = b_start_loc.shape[0]
        for b_idx in range(batch):
            
            seq_len = b_seq_len[b_idx]
            start = b_start_loc[b_idx]
            end = start + seq_len
            single_k = k[start:end]
            single_v = v[start:end]

            block_table = self.get_block_table(int(req_idx[b_idx]))
            block_number = _div_up(seq_len, PagingRequestManager.BLOCK_SIZE)
            last_block_offset = seq_len - (block_number - 1) * PagingRequestManager.BLOCK_SIZE
            for num in range(block_number):
                block_idx = block_table[num]
                offset = last_block_offset if block_number - 1 == num else PagingRequestManager.BLOCK_SIZE
                cache_start = block_idx * PagingRequestManager.BLOCK_SIZE
                kv_start = num * PagingRequestManager.BLOCK_SIZE
                block_k = single_k[kv_start:kv_start+offset]
                block_v = single_v[kv_start:kv_start+offset]
               
                self.mem_manager.key_buffer[layer_num][cache_start:cache_start+offset] = block_k
                self.mem_manager.value_buffer[layer_num][cache_start:cache_start+offset] = block_v
    

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
    
    
