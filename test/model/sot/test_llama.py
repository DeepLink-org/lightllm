import os
import sys
import unittest
from test import test_model_inference

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

class TestLlamaInfer(unittest.TestCase):
    def test_llama_infer(self):
        from lightllm.models.llama.model import LlamaTpPartModel

        test_model_inference(
            world_size=8,
            model_dir="/nvme/nvme2/share/share_data/llama_env/llama-2-7b-chat-hf",
            model_class=LlamaTpPartModel,
            batch_size=1,
            input_len=1024,
            output_len=1024,
            mode=[],
        )
        return


if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_start_method("spawn")
    unittest.main()