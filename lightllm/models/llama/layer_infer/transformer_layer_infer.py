import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from functools import partial

from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev import token_softmax_reducev_fwd

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv
from lightllm.common.basemodel import TransformerLayerInferTpl

from torch.profiler import record_function
import torch._dynamo as dynamo

class LlamaTransformerLayerInfer(TransformerLayerInferTpl):
    """
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.tp_k_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        self._bind_func()

        self.compiled_get_qkv = torch.compile(self.real_get_qkv, backend='ascendgraph', dynamic=False)
        self.compiled_get_o  = torch.compile(self.real_get_o, backend='ascendgraph', dynamic=False)
        self.compiled_ffn = torch.compile(self.real_ffn, backend='ascendgraph', dynamic=False)
        return
    
    def _bind_func(self):
        self._bind_norm()
        self._bind_attention()
        return
    
    def _bind_norm(self):
        self._att_norm = self._att_norm
        self._ffn_norm = self._ffn_norm
        return
    
    def _bind_attention(self):
        self._context_attention_kernel = self._context_attention_kernel
        self._token_attention_kernel = self._token_decode_attention_normal
        self._copy_kv_to_mem_cache = self._copy_kv_to_mem_cache_normal

        return

    def _att_norm(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        return rmsnorm_forward(input, weight=layer_weight.att_norm_weight_, eps=self.eps_)
    
    def _ffn_norm(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        return rmsnorm_forward(input, weight=layer_weight.ffn_norm_weight_, eps=self.eps_)

    def real_get_qkv(self, input, cache_k, cache_v, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        torch.mm(input.view(-1, self.embed_dim_), layer_weight.k_weight_,
                    out=cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_))
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)
        torch.mm(input.view(-1, self.embed_dim_), layer_weight.v_weight_,
                    out=cache_v.view(-1, self.tp_v_head_num_ * self.head_dim_))
        return q, cache_k, cache_v

    def _get_qkv(self, input, cache_k, cache_v, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        q, cache_k, cache_v = self.compiled_get_qkv(input, cache_k, cache_v, infer_state, layer_weight)
        return q, cache_k, cache_v
    
    def _context_attention_kernel(self, q, k, v, infer_state:LlamaInferStateInfo, layer_weight, out=None)->torch.Tensor:
        o_tensor1 = torch.empty_like(q) if out is None else out
        o_tensor = context_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                              k.view(-1, self.tp_k_head_num_, self.head_dim_),
                              v.view(-1, self.tp_v_head_num_, self.head_dim_),
                              o_tensor1.view(-1, self.tp_q_head_num_, self.head_dim_),
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        return o_tensor.view(q.shape)

    def real_get_o(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        o_tensor = torch.mm(input.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)
        return o_tensor

    def _get_o(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        return self.compiled_get_o(input, infer_state, layer_weight)

    def real_ffn(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        gate_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_proj)
        torch.nn.functional.silu(gate_out, inplace=True)
        up_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.up_proj)
        input = None
        ffn1_out = gate_out * up_out
        gate_out, up_out = None, None
        ffn2_out = torch.mm(ffn1_out, layer_weight.down_proj)
        ffn1_out = None
        return ffn2_out

    def _ffn(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        return self.compiled_ffn(input, infer_state, layer_weight)
    
    def _copy_kv_to_mem_cache_normal(self, key_buffer, value_buffer, mem_index, mem_manager):
        destindex_copy_kv(key_buffer, mem_index, mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv(value_buffer, mem_index, mem_manager.value_buffer[self.layer_num_])
        return

    @dynamo.disable
    def _token_decode_attention_normal(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)

        att_m_tensor1 = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")
        with record_function("token_att_fwd"):
            att_m_tensor = token_att_fwd(q.view(calcu_shape1),
                            infer_state.mem_manager.key_buffer[self.layer_num_],
                            att_m_tensor1,
                            infer_state.req_manager.req_to_token_indexs,
                            infer_state.b_req_idx,
                            infer_state.b_start_loc,
                            infer_state.b_seq_len,
                            infer_state.max_len_in_batch)
        
        o_tensor1 = torch.empty_like(q) if out is None else out

        with record_function("token_softmax_reducev_fwd"):
            o_tensor = token_softmax_reducev_fwd(att_m_tensor, 
                                        infer_state.mem_manager.value_buffer[self.layer_num_],
                                        o_tensor1.view(calcu_shape1),
                                        infer_state.req_manager.req_to_token_indexs,
                                        infer_state.b_req_idx,
                                        infer_state.b_start_loc,
                                        infer_state.b_seq_len,
                                        infer_state.max_len_in_batch,
                                        infer_state.other_kv_index)
        return o_tensor

    def ascend_flash_attention_kernel(self, q, k, v, kv_index):
        return torch.ops.lightllm.flash_attention_inference.default(q, k, v, kv_index)

    def real__token_attention_kernel(self, q, infer_state, layer_weight):
        out = self._token_attention_kernel(q, infer_state, layer_weight)
        return out

    def torch_rotary_emb(self, x, cos, sin):
        out = torch.ops.lightllm.rotary_emb.default(x, cos, sin)
        return out

    def pre_process(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
            att_norm_out = torch.ops.lightllm.rms_norm.default(input_embding, layer_weight.att_norm_weight_, self.eps_)

            if infer_state.mem_is_contiguous:
                cache_k = infer_state.mem_manager.key_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_end, :, :]
                cache_v = infer_state.mem_manager.value_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_end, :, :]
            else:
                cache_k = infer_state.key_buffer
                cache_v = infer_state.value_buffer 
            
            # get_qkv
            q = torch.mm(att_norm_out.view(-1, self.embed_dim_), layer_weight.q_weight_)
            q = self.torch_rotary_emb(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
            cache_k = torch.mm(att_norm_out.view(-1, self.embed_dim_), layer_weight.k_weight_)
            cache_k = cache_k.view(-1, self.tp_k_head_num_, self.head_dim_)
            cache_k = self.torch_rotary_emb(cache_k, infer_state.position_cos, infer_state.position_sin)
            cache_v = torch.mm(att_norm_out.view(-1, self.embed_dim_), layer_weight.v_weight_)
            cache_v = cache_v.view(-1, self.tp_v_head_num_, self.head_dim_)
                
            self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
            return q
    
    def post_process(self, o, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
            o = torch.mm(o.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)
            input_embding = input_embding + o.view(-1, self.embed_dim_)
            
            input1 = torch.ops.lightllm.rms_norm.default(input_embding, layer_weight.att_norm_weight_, self.eps_)

            gate_out = torch.mm(input1.view(-1, self.embed_dim_), layer_weight.gate_proj)
            gate_out = torch.nn.functional.silu(gate_out)
            up_out = torch.mm(input1.view(-1, self.embed_dim_), layer_weight.up_proj)
            input1 = None
            ffn1_out = gate_out * up_out
            gate_out, up_out = None, None
            ffn2_out = torch.mm(ffn1_out, layer_weight.down_proj)
            ffn1_out = None
            input_embding = input_embding + ffn2_out.view(-1, self.embed_dim_)
            return input_embding

    def full_token_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight, kv_index):
            q = self.pre_process(input_embding, infer_state, layer_weight)
            o = torch.ops.lightllm.flash_attention_inference.default(q, infer_state.mem_manager.key_buffer[self.layer_num_], infer_state.mem_manager.value_buffer[self.layer_num_], kv_index)
            out = self.post_process(o, input_embding, infer_state, layer_weight)
            return out
