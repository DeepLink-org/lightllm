import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import ReqManager
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight

class LlamaInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.other_kv_index = None
        self.layer0_weight:LlamaTransformerLayerWeight = None
    
    def init_some_extra_state(self, model, input_ids : torch.Tensor):
        self.init_tmp_out()
        if self.is_prefill:
            position_ids = torch.from_numpy(np.concatenate([np.arange(0, self.max_len_in_batch)
                                            for i in range(self.batch_size)], axis=0)).cuda()
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(1, position_ids.shape[0], 1, -1).repeat(1,1,1,2)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(1, position_ids.shape[0], 1, -1).repeat(1,1,1,2)
            position_ids = None
        else:
            position_ids = self.b_seq_len - 1
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(1, self.b_seq_len.shape[0], 1, -1).repeat(1,1,1,2)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(1, self.b_seq_len.shape[0], 1, -1).repeat(1,1,1,2)
            self.other_kv_index = self.req_manager.req_to_token_indexs[self.b_req_idx[0], 0].item()
            # b_loc[0, max_len_in_batch - 1].item()
        return
    
    def init_tmp_out(self):
        if self.is_prefill:
            self.rms_norm_out = torch.empty([self.total_token_num, self.embed_dim], dtype=torch.float16, device='cuda')
            self.matmul_all_reduce_out = torch.empty([self.total_token_num, self.embed_dim], dtype=torch.float16, device='cuda')
            self.ffn1_out = torch.empty([self.total_token_num, self.layer0_weight.up_proj.shape[1]], dtype=torch.float16, device='cuda')
            self.ffn2_out = torch.empty([self.total_token_num, self.embed_dim], dtype=torch.float16, device='cuda')
            self.attention_out = torch.empty([self.total_token_num, self.tp_q_head_num, self.head_dim], dtype=torch.float16, device='cuda')
            self.q = torch.empty([self.total_token_num, self.tp_q_head_num, self.head_dim], dtype=torch.float16, device='cuda')
        else:
            self.rms_norm_out = torch.empty([self.b_seq_len.shape[0], self.embed_dim], dtype=torch.float16, device='cuda')
            self.matmul_all_reduce_out = torch.empty([self.b_seq_len.shape[0], self.embed_dim], dtype=torch.float16, device='cuda')
            self.ffn1_out = torch.empty([self.b_seq_len.shape[0], self.layer0_weight.up_proj.shape[1]], dtype=torch.float16, device='cuda')
            self.ffn2_out = torch.empty([self.b_seq_len.shape[0], self.embed_dim], dtype=torch.float16, device='cuda')
            self.attention_out = torch.empty([self.b_seq_len.shape[0], self.tp_q_head_num, self.head_dim], dtype=torch.float16, device='cuda')
            self.q = torch.empty([self.b_seq_len.shape[0], self.tp_q_head_num, self.head_dim], dtype=torch.float16, device='cuda')

        self.inv_rms = torch.empty(list(self.rms_norm_out.shape[:-1]) + [1], dtype=torch.float32, device='cuda')


   
        
