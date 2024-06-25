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


import logging
# torch._logging.set_logs(dynamo=logging.DEBUG, output_code=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_llama2_infer():
    from lightllm.models.llama.model import LlamaTpPartModel
    #model_dir = "/tzy/llama-2-70b-chat-hf/"
    model_dir = "/data2/share_data/llama_model_data/llama-2-70b-chat-hf"
    test_model_inference(world_size=4,
                         model_dir=model_dir,
                         model_class=LlamaTpPartModel,
                         batch_size=2,
                         input_len=16,
                         output_len=4,
                         mode=[],
                         dynamic_compiler=True)
    return

if __name__ == '__main__':
    test_llama2_infer()
