import os
import sys
import unittest

import torch._dynamo
import torch
import torch_dipu
import logging

torch._dynamo.config.suppress_errors = False
torch._dynamo.config.cache_size_limit = 3000

# torch._logging.set_logs(dynamo = logging.DEBUG)

from model_infer import test_model_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_llama2_infer():
    from lightllm.models.llama.model import LlamaTpPartModel
    test_model_inference(world_size=1,
                         model_dir="/data/share_data/llama_model_data/llama-2-7b-chat-hf",
                         model_class=LlamaTpPartModel,
                         batch_size=1,
                         input_len=16,
                         output_len=4,
                         mode=[])
    return

if __name__ == '__main__':
    test_llama2_infer()
