import os
import sys
import unittest

import torch._dynamo
import torch
import torch_dipu

import torch.fx.graph_module
from model_infer import test_model_inference

torch._dynamo.config.suppress_errors = False
torch._dynamo.config.cache_size_limit = 3000

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_llama2_infer():
    from lightllm.models.llama.model import LlamaTpPartModel
    model_dir = "/data/share_data/llama_model_data/llama-2-7b-chat-hf"
    test_model_inference(world_size=1,
                         model_dir=model_dir,
                         model_class=LlamaTpPartModel,
                         batch_size=1,
                         input_len=4,
                         output_len=2,
                         max_prompt_size=64,
                         is_padding=False,
                         mode=[])
    return

if __name__ == '__main__':
    test_llama2_infer()
