import torch
import torch.functional as F
import torch.distributed as dist
import torch._dynamo as dynamo
import numpy as np
from typing import Tuple
from functools import partial
from torch.profiler import record_function

from dicp.vendor.AscendGraph import ext_ops
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev import token_softmax_reducev_fwd

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv
from lightllm.common.basemodel import TransformerLayerInferTpl


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

        self.opt_context_pre_process = torch.compile(self.context_pre_process, backend='ascendgraph', dynamic=False)
        self.opt_context_post_process = torch.compile(self.context_post_process, backend='ascendgraph', dynamic=False)
        self.opt_token_pre_process = torch.compile(self.token_pre_process, backend='ascendgraph', dynamic=False)
        self.opt_token_post_process = torch.compile(self.token_post_process, backend='ascendgraph', dynamic=False)
        self.opt_ascend_prompt_attention_inference = torch.compile(self.ascend_prompt_attention_kernel, backend='ascendgraph', dynamic=False)
        self.opt_ascend_incre_attention_kernel = torch.compile(self.ascend_incre_attention_kernel, backend='ascendgraph', dynamic=False)
        self.opt_full_context_attention= torch.compile(self.full_context_attention, backend='ascendgraph', dynamic=False)
        self.opt_full_token_attention= torch.compile(self.full_token_attention, backend='ascendgraph', dynamic=False)

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
    
    def _copy_kv_to_mem_cache_normal(self, key_buffer, value_buffer, mem_index, mem_manager):
        destindex_copy_kv(key_buffer, mem_index, mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv(value_buffer, mem_index, mem_manager.value_buffer[self.layer_num_])
        return
    
    @record_function('transformer_context_attention_kernel')
    def _context_attention_kernel(self, q, k, v, infer_state:LlamaInferStateInfo, layer_weight, out=None)->torch.Tensor:
        o_tensor1 = torch.empty_like(q) if out is None else out
        o_tensor = context_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                              k.view(-1, self.tp_k_head_num_, self.head_dim_),
                              v.view(-1, self.tp_v_head_num_, self.head_dim_),
                              o_tensor1.view(-1, self.tp_q_head_num_, self.head_dim_),
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch,
                              infer_state.masks,
                              infer_state.is_padding)
        return o_tensor
    
    @record_function('transformer_token_decode_attention_normal')
    def _token_decode_attention_normal(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        
        att_m_tensor1 = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")

        att_m_tensor = token_att_fwd(q.view(calcu_shape1),
                        infer_state.mem_manager.key_buffer[self.layer_num_],
                        att_m_tensor1,
                        infer_state.req_manager.req_to_token_indexs,
                        infer_state.b_req_idx,
                        infer_state.b_start_loc,
                        infer_state.b_seq_len,
                        infer_state.max_len_in_batch)
        
        o_tensor1 = torch.empty_like(q) if out is None else out

        o_tensor = token_softmax_reducev_fwd(att_m_tensor, 
                                infer_state.mem_manager.value_buffer[self.layer_num_],
                                o_tensor1.view(calcu_shape1),
                                infer_state.is_padding,
                                infer_state.masks,
                                infer_state.req_manager.req_to_token_indexs,
                                infer_state.b_req_idx,
                                infer_state.b_start_loc,
                                infer_state.b_seq_len,
                                infer_state.max_len_in_batch,
                                infer_state.other_kv_index)
        return o_tensor.view(q.shape)
    
    def torch_rotary_emb(self, x, cos, sin):
        out = torch.ops.lightllm.rotary_emb.default(x, cos, sin)
        return out
    
    def ascend_prompt_attention_kernel(self, q, k, v, seqlen, num_head , head_dim):
        return torch.ops.lightllm.prompt_attention_inference.default(q, k, v, seqlen, num_head, head_dim)
    
    def ascend_incre_attention_kernel(self, q, k, v, int_index_list, max_seq_length):
        return torch.ops.lightllm.flash_attention_inference.default(q, k, v, int_index_list, max_seq_length)

    def context_pre_process(self, input_embdings, infer_state, layer_weight):
        # att_norm
        input1 = torch.ops.lightllm.rms_norm.default(input_embdings.float(), layer_weight.att_norm_weight_.float(), self.eps_).half()
        
        # pre_cache_kv
        cache_k = infer_state.mem_manager.key_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_end, :, :]
        cache_v = infer_state.mem_manager.value_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_end, :, :]

        # get_qkv
        q = torch.mm(input1.view(-1, self.embed_dim_), layer_weight.q_weight_)
        q = self.torch_rotary_emb(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)

        cache_k = torch.mm(input1.view(-1, self.embed_dim_), layer_weight.k_weight_).view(-1, self.tp_k_head_num_, self.head_dim_)
        cache_k = self.torch_rotary_emb(cache_k, infer_state.position_cos, infer_state.position_sin)

        cache_v = torch.mm(input1.view(-1, self.embed_dim_), layer_weight.v_weight_).view(-1, self.tp_v_head_num_, self.head_dim_)

        # post cache_kv
        infer_state.mem_manager.key_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_end] = cache_k
        infer_state.mem_manager.value_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_end] = cache_v

        return q, cache_k, cache_v

    def context_post_process(self, input_embdings, out, layer_weight):
        # get_o
        o = torch.mm(out.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)

        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(o.view(-1, self.embed_dim_))

        # ffn_norm
        input1 = torch.ops.lightllm.rms_norm.default(input_embdings.float(), layer_weight.ffn_norm_weight_.float(), self.eps_).half()

        # ffn
        gate_out = torch.mm(input1.view(-1, self.embed_dim_), layer_weight.gate_proj)
        torch.nn.functional.silu(gate_out, inplace=True)
        up_out = torch.mm(input1.view(-1, self.embed_dim_), layer_weight.up_proj)
        input1 = None
        ffn1_out = gate_out * up_out
        gate_out, up_out = None, None
        ffn2_out = torch.mm(ffn1_out, layer_weight.down_proj)
        ffn1_out = None

        # if self.world_size_ > 1:
        #     dist.all_reduce(ffn2_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn2_out.view(-1, self.embed_dim_))

        return input_embdings
    
    def full_context_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        q, k, v = self.opt_context_pre_process(input_embding, infer_state, layer_weight)

        q = q.view(-1, self.tp_q_head_num_, self.head_dim_)

        batch, num_head, head_dim = infer_state.b_start_loc.shape[0], q.shape[1], q.shape[2]
        seqlen = infer_state.b_seq_len

        out = self.opt_ascend_prompt_attention_inference(q.view(batch, -1, num_head * head_dim), 
                                                            k.view(batch, -1, num_head * head_dim), 
                                                            v.view(batch, -1, num_head * head_dim), 
                                                            seqlen, num_head, head_dim)

        out = out.view(-1, self.tp_q_head_num_ * self.head_dim_)

        out = self.opt_context_post_process(input_embding, out, layer_weight)

        return out
    
    def token_pre_process(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        # att_norm
        att_norm_out = torch.ops.lightllm.rms_norm.default(input_embding.float(), layer_weight.att_norm_weight_.float(), self.eps_).half()

        # pre_kv
        cache_k = infer_state.key_buffer
        cache_v = infer_state.value_buffer 

        # get_qkv
        tmp_res = torch.mm(att_norm_out.view(-1, self.embed_dim_), layer_weight.qkv_weight_ )
        (q, cache_k, cache_v) = torch.ops.aten.split(tmp_res, self.embed_dim_, dim=1)

        q = self.torch_rotary_emb(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        cache_k = cache_k.view(-1, self.tp_k_head_num_, self.head_dim_)
        cache_k = self.torch_rotary_emb(cache_k, infer_state.position_cos, infer_state.position_sin)
        cache_v = cache_v.view(-1, self.tp_v_head_num_, self.head_dim_)

        # post cache_kv
        start_index = infer_state.int_index_list[0]
        end_index = infer_state.int_index_list[0] + 1
        infer_state.mem_manager.key_buffer[self.layer_num_][start_index:end_index] = cache_k
        infer_state.mem_manager.value_buffer[self.layer_num_][start_index:end_index] = cache_v

        return q
    
    def token_post_process(self, o, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        # get_o
        o = torch.mm(o.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)

        input_embding.add_(o.view(-1, self.embed_dim_))
        
        # ffn_norm
        input1 = torch.ops.lightllm.rms_norm.default(input_embding.float(), layer_weight.ffn_norm_weight_.float(), self.eps_).half()
        
        tmp_res = torch.mm(input1.view(-1, self.embed_dim_), layer_weight.gate_up_proj )
        (gate_out, up_out) = torch.ops.aten.split(tmp_res, layer_weight.gate_up_proj.size()[1] // 2, dim=1)
        
        gate_out = torch.nn.functional.silu(gate_out)

        input1 = None
        ffn1_out = gate_out * up_out
        gate_out, up_out = None, None
        ffn2_out = torch.mm(ffn1_out, layer_weight.down_proj)
        ffn1_out = None

        input_embding.add_(ffn2_out.view(-1, self.embed_dim_))

        return input_embding

    def full_token_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight, current_len, max_seq_length):
        q = self.opt_token_pre_process(input_embding, infer_state, layer_weight)
        o = self.opt_ascend_incre_attention_kernel(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.mem_manager.key_buffer[self.layer_num_], infer_state.mem_manager.value_buffer[self.layer_num_], infer_state.int_index_list_t, max_seq_length)
        out = self.opt_token_post_process(o, input_embding, infer_state, layer_weight)
        return out
