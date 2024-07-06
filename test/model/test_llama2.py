import os
import sys
import unittest
import torch
import torch_dipu
from model_infer import test_model_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestLlama2Infer(unittest.TestCase):

    def test_llama2_infer(self):
        from lightllm.models.llama.model import LlamaTpPartModel
        test_model_inference(world_size=1, 
                             model_dir="/data/share/llama2/Llama-2-7b-hf/", 
                             model_class=LlamaTpPartModel, 
                             batch_size=1, 
                             input_len=256, 
                             output_len=0,
                             mode=[])
        return


if __name__ == '__main__':
    unittest.main()