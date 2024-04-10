import os
import sys
import unittest
import torch
import torch_dipu
import deeplink_ext.patch_lightllm
import lightllm
from model_infer import test_model_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestLlama2Infer(unittest.TestCase):

    def test_llama2_infer(self):
        from lightllm.models.llama.model import LlamaTpPartModel
        test_model_inference(world_size=8, 
                             model_dir="/data01/zhaochaoxing/tools/llama-2-70b-chat-hf", 
                             model_class=LlamaTpPartModel, 
                             batch_size=1, 
                             input_len=256, 
                             output_len=4,
                             mode=[])
        return


if __name__ == '__main__':
    unittest.main()