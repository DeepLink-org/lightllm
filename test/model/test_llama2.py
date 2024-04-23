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
                             model_dir="/mnt/cache/share_data/PAT/datasets/Llama-2-70b-hf/", 
                             model_class=LlamaTpPartModel, 
                             batch_size=8, 
                             input_len=256, 
                             output_len=6,
                             mode=[])
        # test_model_inference(world_size=1, 
        #                      model_dir="/data2/share_data/llama_model_data/llama-2-7b-chat-hf", 
        #                      model_class=LlamaTpPartModel, 
        #                      batch_size=8, 
        #                      input_len=256, 
        #                      output_len=3,
        #                      mode=[])
        return


if __name__ == '__main__':
    unittest.main()