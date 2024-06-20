import torch
import torch.functional as F
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import numpy as np
from lightllm.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight
from lightllm.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from einops import rearrange
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.common.basemodel import PostLayerInferTpl
from lightllm.utils.infer_utils import mark_cost_time

from torch.profiler import record_function


class LlamaPostLayerInfer(PostLayerInferTpl):
    """
    """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]

        self.opt_token_forward = torch.compile(self.token_forward, backend='ascendgraph', dynamic=True)
        self.opt_dicp_token_forward = torch.compile(self.dicp_token_forward, backend='ascendgraph', dynamic=True)

        return
    
    def _norm(self, input, final_norm_weight_) -> torch.Tensor:
        return input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + self.eps_) * final_norm_weight_
    
    def _slice_get_last_input(self, input_embdings, infer_state: LlamaInferStateInfo):
        if infer_state.is_splitfuse:
            # for SplitFuse
            batch_size = infer_state.batch_size
            last_input = torch.empty((batch_size, self.embed_dim_), device=input_embdings.device, dtype=torch.float16)
            tmp_ = torch.cat([torch.ones(infer_state.decode_req_num, dtype=torch.int32, device="cuda"), infer_state.prefill_b_split_seq_len], dim=0)
            last_index = torch.cumsum(tmp_, dim=0, dtype=torch.long) - 1
            last_input[:, :] = input_embdings[last_index, :]
            return last_input, batch_size
        
        if not infer_state.is_splitfuse and infer_state.is_prefill and not infer_state.return_all_prompt_logprobs:
            batch_size = infer_state.batch_size
            last_input = torch.empty((batch_size, self.embed_dim_), device=input_embdings.device, dtype=torch.float16)
            last_index = torch.cumsum(infer_state.b_seq_len, dim=0, dtype=torch.long) - 1
            last_input[:, :] = input_embdings[last_index, :]
            return last_input, batch_size

        if not infer_state.is_splitfuse and infer_state.is_prefill and infer_state.return_all_prompt_logprobs:
            total_tokens = infer_state.total_token_num
            return input_embdings, total_tokens
        
        if not infer_state.is_splitfuse and not infer_state.is_prefill:
            batch_size = infer_state.batch_size
            return input_embdings[-batch_size:, :], batch_size

    def soft_max(self, data):
        return torch.softmax(data.permute(1, 0).float(), dim=-1)

    def post_token_forward(self, lm_head_weight_, final_norm_weight_, last_input, token_num, default_pg, return_logics=False):
        last_input = self._norm(last_input, final_norm_weight_)
        last_input = rearrange(last_input, "batch embed_dim -> embed_dim batch").contiguous().reshape(-1, token_num)
        logic_batch = torch.mm(lm_head_weight_, last_input)

        last_input = None
        if self.world_size_ == 1:
            gather_data = logic_batch
        else:
            # gather_data = funcol.all_gather_tensor(logic_batch, 0, default_pg)
            tag, rankset, group_size = funcol._expand_group(default_pg, '')
            tensor = torch.ops.c10d_functional.all_gather_into_tensor(logic_batch, tag, rankset, group_size)  # type: ignore[attr-defined]
            gather_data = funcol._maybe_wrap_tensor(tensor)
            
            
        logic_batch = None

        if not return_logics:
            prob_out = self.soft_max(gather_data)
            gather_data = None
            return prob_out
        else:
            # here
            ans_logics = gather_data.permute(1, 0).float()
            gather_data = None
            return ans_logics

    def token_forward(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight, default_pg, return_logics=False):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        out = self.post_token_forward(layer_weight.lm_head_weight_, layer_weight.final_norm_weight_, last_input, token_num, default_pg, return_logics)
        return out
    
    def dicp_token_forward(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight, batch_size, default_pg, return_logics=False):
        last_input = input_embdings[-batch_size:, :]
        token_num = batch_size
        out = self.post_token_forward(layer_weight.lm_head_weight_, layer_weight.final_norm_weight_, last_input, token_num, default_pg, return_logics)
        return out

    @mark_cost_time("splitfuse post forward")
    def splitfuse_forward(self, input_embdings, infer_state: SplitFuseInferStateInfo, layer_weight: BaseLayerWeight, return_logics=False):
        return self.token_forward(input_embdings, infer_state, layer_weight, return_logics=return_logics)
