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
                         model_dir="/data/share_data/llama_model_data/llama-2-7b-chat-hf",
                         model_class=LlamaTpPartModel,
                         batch_size=1,
                         input_len=256,
                         output_len=6,
                         mode=[])
    return

if __name__ == '__main__':
    # import torch_npu
    # with torch_npu.npu.profile(profiler_result_path="./result"):
    test_llama2_infer()
