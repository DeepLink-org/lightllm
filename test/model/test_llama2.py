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

# monkey patch for pytorch
import importlib
tmp_variable_torch_module = importlib.import_module("torch._dynamo.variables.torch")
tmp_torch_variable = getattr(tmp_variable_torch_module, "TorchVariable")
origin_torch_variable_python_type = getattr(tmp_torch_variable, "python_type")
def new_torch_variable_python_type(self):
    if isinstance(self.value, torch.device):
        return type(self.value)
    else:
        return origin_torch_variable_python_type(self)
setattr(tmp_torch_variable, "python_type", new_torch_variable_python_type)

torch._dynamo.config.dynamic_shapes = True
torch._dynamo.config.assume_static_by_default = True
# torch._dynamo.config.specialize_int = True


import logging
# torch._logging.set_logs(dynamo=logging.DEBUG, output_code=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_llama2_infer():
    from lightllm.models.llama.model import LlamaTpPartModel
    #model_dir = "/tzy/llama-2-70b-chat-hf/"
    model_dir = "/data2/pandaoxin/workspace/llama-model-data/llama-2-7b-chat-hf"
    test_model_inference(world_size=1,
                         model_dir=model_dir,
                         model_class=LlamaTpPartModel,
                         batch_size=1,
                         input_len=16,
                         output_len=4,
                         mode=[],
                         dynamic_compiler=True)
    return

if __name__ == '__main__':
    test_llama2_infer()
