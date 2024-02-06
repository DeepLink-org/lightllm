import torch
import torch.distributed as dist
from ..transformer_layer_infer import TransformerLayerInfer
from ...infer_struct import InferStateInfo
from ...splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd

from typing import Tuple

from torch.profiler import record_function
class TransformerLayerInferTpl(TransformerLayerInfer):
    """
    """
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        # need to set by subclass
        self.eps_ = 1e-5 
        self.tp_q_head_num_ = -1
        self.tp_k_head_num_ = -1
        self.tp_v_head_num_ = -1
        self.tp_o_head_num_ = -1
        self.head_dim_ = -1
        self.embed_dim_ = -1
        self.compiled_pre_kernel = torch.compile(self.pre_kernel, backend='ascendgraph')
        self.compiled_post_kernel = torch.compile(self.post_kernel, backend='ascendgraph')
        return
    
    def _att_norm(self, input, infer_state:InferStateInfo, layer_weight)->torch.Tensor:
        raise Exception("need to impl")
    
    def _ffn_norm(self, input, infer_state:InferStateInfo, layer_weight)->torch.Tensor:
        raise Exception("need to impl")
    
    def _pre_cache_kv(self, infer_state:InferStateInfo, layer_weight)->Tuple[torch.Tensor, torch.Tensor]:
        if infer_state.mem_is_contiguous:
            cache_k = infer_state.mem_manager.key_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_end, :, :]
            cache_v = infer_state.mem_manager.value_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_end, :, :]
        else:
            cache_k = infer_state.key_buffer
            cache_v = infer_state.value_buffer 
        return cache_k, cache_v

    def _get_qkv(self, input, cache_k, cache_v, infer_state:InferStateInfo, layer_weight)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise Exception("need to impl")
    
    def _post_cache_kv(self, cache_k, cache_v, infer_state:InferStateInfo, layer_weight):
        mem_manager = infer_state.mem_manager
        if not infer_state.mem_is_contiguous:
            self._copy_kv_to_mem_cache(cache_k, cache_v, infer_state.mem_index, mem_manager)
            return
    
    def _copy_kv_to_mem_cache(self, key_buffer, value_buffer, mem_index, mem_manager):
        destindex_copy_kv(key_buffer, mem_index, mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv(value_buffer, mem_index, mem_manager.value_buffer[self.layer_num_])
        return
    
    def _context_attention_kernel(self, q, k, v, infer_state:InferStateInfo, layer_weight, out=None)->torch.Tensor:
        raise Exception("need to impl")
    
    def _token_attention_kernel(self, q, infer_state:InferStateInfo, layer_weight, out=None)->torch.Tensor:
        raise Exception("need to impl")
    
    def _splitfuse_attention_kernel(self, q, infer_state:SplitFuseInferStateInfo, layer_weight, out=None)->torch.Tensor:
        raise Exception("need to impl")

    def _get_o(self, input, infer_state:InferStateInfo, layer_weight)->torch.Tensor:
        raise Exception("need to impl")

    def _ffn(self, input, infer_state:InferStateInfo, layer_weight)->torch.Tensor:
        raise Exception("need to impl")


    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_attention(self, input_embding, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v  = self._get_qkv(input1, cache_k, cache_v, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return

    def post_kernel(self, input_embdings, out, layer_weight):
        # get_o
        o = torch.mm(out.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)

        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(o.view(-1, self.embed_dim_))

        # ffn_norm
        input1 = input_embdings * torch.rsqrt(input_embdings.pow(2).mean(-1, keepdim=True) + self.eps_) * layer_weight.ffn_norm_weight_

        # ffn
        gate_out = torch.mm(input1.view(-1, self.embed_dim_), layer_weight.gate_proj)
        torch.nn.functional.silu(gate_out, inplace=True)
        up_out = torch.mm(input1.view(-1, self.embed_dim_), layer_weight.up_proj)
        input1 = None
        ffn1_out = gate_out * up_out
        gate_out, up_out = None, None
        ffn2_out = torch.mm(ffn1_out, layer_weight.down_proj)
        ffn1_out = None

        input1 = None
        # if self.world_size_ > 1:
        #     dist.all_reduce(ffn2_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn2_out.view(-1, self.embed_dim_))

        return

    def pre_kernel(self, input_embdings, out, cache_k, cache_v, infer_state: InferStateInfo, layer_weight_t, layer_weight):
        # get_o
        o = torch.mm(out.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight_t.o_weight_)

        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(o.view(-1, self.embed_dim_))

        # ffn_norm
        input1 = input_embdings * torch.rsqrt(input_embdings.pow(2).mean(-1, keepdim=True) + self.eps_) * layer_weight_t.ffn_norm_weight_

        # ffn
        gate_out = torch.mm(input1.view(-1, self.embed_dim_), layer_weight_t.gate_proj)
        torch.nn.functional.silu(gate_out, inplace=True)
        up_out = torch.mm(input1.view(-1, self.embed_dim_), layer_weight_t.up_proj)
        ffn1_out = gate_out * up_out
        gate_out, up_out = None, None
        ffn2_out = torch.mm(ffn1_out, layer_weight_t.down_proj)
        ffn1_out = None

        input1 = None
        # if self.world_size_ > 1:
        #     dist.all_reduce(ffn2_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn2_out.view(-1, self.embed_dim_))

        # att_norm
        input1 = input_embdings * torch.rsqrt(input_embdings.pow(2).mean(-1, keepdim=True) + self.eps_) * layer_weight.att_norm_weight_

        # get_qkv
        q = torch.mm(input1.view(-1, self.embed_dim_), layer_weight.q_weight_)
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        torch.mm(input1.view(-1, self.embed_dim_), layer_weight.k_weight_,
                    out=cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_))
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)
        torch.mm(input1.view(-1, self.embed_dim_), layer_weight.v_weight_,
                    out=cache_v.view(-1, self.tp_v_head_num_ * self.head_dim_))

        return q, cache_k, cache_v

    @record_function("first_context_forward")
    def first_context_forward(self, input_embdings, cache_k, cache_v, infer_state: InferStateInfo, layer_weight):
        with record_function("att_norm"):
            input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        with record_function("get_qkv"):
            q, cache_k, cache_v  = self._get_qkv(input1, cache_k, cache_v, infer_state, layer_weight)
        with record_function("post_cache_kv"):
            self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        with record_function("context_attention_kernel"):
            out = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        return input_embdings, out, layer_weight

    @record_function("tmp_context_forward")
    def tmp_context_forward(self, input_embdings, out, cache_k, cache_v, infer_state: InferStateInfo, layer_weight, layer_weight_t: None, is_last_layer: bool=False):
        with record_function("compiled_pre_kernel"):
            q, cache_k, cache_v = self.compiled_pre_kernel(input_embdings, out, cache_k, cache_v, infer_state, layer_weight_t, layer_weight)
        with record_function("post_cache_kv"):
            self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        with record_function("context_attention_kernel"):
            out = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        if is_last_layer:
            with record_function("compiled_post_kernel"):
                self.compiled_post_kernel(input_embdings, out, layer_weight)
                return input_embdings
        return input_embdings, out, layer_weight

    @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return

    # this impl dont to use @mark_cost_time
    def _token_attention(self, input_embding, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv(input1, cache_k, cache_v, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return

    # this impl dont to use @mark_cost_time
    def _token_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return
    
    # @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _splitfuse_attention(self, input_embding, infer_state: SplitFuseInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v  = self._get_qkv(input1, cache_k, cache_v, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._splitfuse_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return

    # @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _splitfuse_ffn(self, input_embdings, infer_state: SplitFuseInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return
    
    @record_function('transformer_context_forward')
    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        self._context_attention(input_embdings,
                                      infer_state,
                                      layer_weight=layer_weight)
        self._context_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings

    @record_function('transformer_token_forward')
    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        self._token_attention(input_embdings,
                                    infer_state,
                                    layer_weight=layer_weight)
        self._token_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings
    
    def splitfuse_forward(self, input_embdings, infer_state: SplitFuseInferStateInfo, layer_weight):
        self._splitfuse_attention(input_embdings,
                            infer_state,
                            layer_weight=layer_weight)
        self._splitfuse_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings
