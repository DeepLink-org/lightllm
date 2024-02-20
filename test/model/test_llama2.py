import os
import sys
import unittest

# import torch._dynamo
import torch

# torch._dynamo.config.suppress_errors = False
# torch._dynamo.config.cache_size_limit = 3000

from model_infer import test_model_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_llama2_infer():
    from lightllm.models.llama.model import LlamaTpPartModel
    test_model_inference(world_size=1,
                         model_dir="/tzy/llama-2-7b-chat-hf",
                         model_class=LlamaTpPartModel,
                         batch_size=1,
                         input_len=128,
                         output_len=128,
                         mode=[])
    return

if __name__ == '__main__':
    test_llama2_infer()
